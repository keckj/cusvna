import os, sys, enum, collections

import cupy as cp
import numpy as np
import optix as ox

import glfw
import imgui

from utils import logger as cusvna_logger, without_fast_math

script_dir = os.path.dirname(os.path.abspath(__file__))
sutil_directory = os.path.join(script_dir, '..', 'ext', 'python-optix', 'examples')

if not os.path.isdir(sutil_directory):
    msg = "Cannot find python-optix submodule, did you clone the repository recursively ?"
    cusvna_logger.fatal(msg)
    raise RuntimeError(msg)

sys.path.insert(0, sutil_directory)

from sutil.gui import init_ui, static_vars
from sutil.gl_display import GLDisplay
from sutil.trackball import Trackball, TrackballViewMode
from sutil.cuda_output_buffer import CudaOutputBuffer, CudaOutputBufferType, BufferImageFormat

DEBUG=False

if DEBUG:
    exception_flags=ox.ExceptionFlags.DEBUG | ox.ExceptionFlags.TRACE_DEPTH | ox.ExceptionFlags.STACK_OVERFLOW
    debug_level = ox.CompileDebugLevel.FULL
    opt_level = ox.CompileOptimizationLevel.LEVEL_0
    validation_mode = True
else:
    exception_flags=ox.ExceptionFlags.NONE
    debug_level = ox.CompileDebugLevel.MINIMAL
    opt_level = ox.CompileOptimizationLevel.LEVEL_3
    validation_mode = False

#------------------------------------------------------------------------------
# Scene
#------------------------------------------------------------------------------

class Shape(enum.IntEnum):
    """Must match the Shape enum of cuda/levelset.cu."""
    FLOOR       = 0
    GYROID      = 1
    METABALLS   = 2
    TWISTED_BOX = 3
    KUMMER      = 4
    BARTH       = 5
    HEART       = 6
    TETRAHEDRAL = 7

class RenderingMode(enum.IntEnum):
    SHADED  = 0
    NORMALS = 1
    STEPS   = 2

class MarchingMethod(enum.IntEnum):
    """See march() in cuda/levelset.cu."""
    SIGN_TEST   = 0
    ROOT_WINDOW = 1

# must match the Object struct of cuda/levelset.cu
object_dtype = np.dtype([
    ('center', '3f4'),
    ('scale',  'f4'),
    ('spin',   'f4'),
    ('shape',  'i4'),
    ('phase',  'f4'),
    ('pad',    'f4'),
])

floor_y = -1.5

def make_scene():
    """A gyroid ball surrounded by a ring of animated level sets, standing above a checkered floor."""
    # around the gyroid, starting in front of the camera
    ring = [Shape.HEART, Shape.KUMMER, Shape.BARTH, Shape.TETRAHEDRAL, Shape.TWISTED_BOX, Shape.METABALLS]
    ring_radius, ring_scale = 4.0, 0.95

    objects = np.zeros(2 + len(ring), dtype=object_dtype)
    objects[0] = ((0.0, floor_y, 0.0), 0.0, 0.0, Shape.FLOOR, 0.0, 0.0)
    objects[1] = ((0.0, floor_y + 1.45, 0.0), 1.4, 0.25, Shape.GYROID, 0.0, 0.0)
    for i, shape in enumerate(ring):
        angle = np.pi/2 + 2*np.pi*(i + 0.5)/len(ring)
        center = (ring_radius*np.cos(angle), floor_y + ring_scale + 0.1, -ring_radius*np.sin(angle))
        spin = 0.5 * (-1)**i
        objects[2+i] = (center, ring_scale, spin, shape, 1.7*i, 0.0)
    return objects

def compute_bboxes(objects):
    bboxes = np.empty((objects.size, 6), dtype=np.float32)
    extent = objects['scale'].reshape(-1, 1)
    bboxes[:, :3] = objects['center'] - extent
    bboxes[:, 3:] = objects['center'] + extent
    floor = (objects['shape'] == Shape.FLOOR)
    bboxes[floor] = (-50.0, floor_y - 1e-3, -50.0, 50.0, floor_y + 1e-3, 50.0)
    return bboxes

objects = make_scene()
d_objects = cp.asarray(objects.view(np.uint8))

#------------------------------------------------------------------------------
# Local types
#------------------------------------------------------------------------------

class Params:
    _params = collections.OrderedDict([
            ('frame_buffer',   'u8'),
            ('width',          'u4'),
            ('height',         'u4'),
            ('ray_marching_steps', 'u4'),
            ('ray_marching_threshold', 'f4'),
            ('eye',            '3f4'),
            ('u',              '3f4'),
            ('v',              '3f4'),
            ('w',              '3f4'),
            ('trav_handle',    'u8'),
            ('rendering_mode', 'u4'),
            ('subframe_index', 'i4'),
            ('time',           'f4'),
            ('marching_method', 'u4'),
            ('samples_per_pixel', 'u4'),
            ('pad',            'u4'),
            ('accum_buffer',   'u8'),
        ])

    def __init__(self):
        self.handle = ox.LaunchParamsRecord(names=tuple(self._params.keys()),
                                            formats=tuple(self._params.values()))

    def __getattribute__(self, name):
        if name in Params._params.keys():
            return self.__dict__['handle'][name][0]
        else:
            return super().__getattribute__(name)

    def __setattr__(self, name, value):
        if name in Params._params.keys():
            self.handle[name] = value
        elif name in {'handle'}:
            super().__setattr__(name, value)
        else:
            raise AttributeError(name)

    def __str__(self):
        return '\n'.join(f'{k}:  {self.handle[k]}' for k in self._params)


class DynamicGeometryState:
    __slots__ = ['params', 'time', 'ctx', 'module', 'pipeline', 'pipeline_opts',
            'raygen_grp', 'miss_grps', 'hit_grp', 'sbt', 'paused', 'speed', 'accum_buffer',
            'generate_vertices_kernel', 'd_temp_vertices', 'last_exploding_sphere_rebuild_time',
            'gas', 'ias',
            'trackball', 'camera_changed', 'mouse_button', 'resize_dirty', 'minimized']

    def __init__(self):
        for slot in self.__slots__:
            setattr(self, slot, None)
        self.params = Params()

        self.trackball = Trackball()
        self.camera_changed = True
        self.mouse_button = -1
        self.resize_dirty = False
        self.minimized = False
        self.paused = False
        self.speed = 1.0

    @property
    def camera(self):
        return self.trackball.camera

    @property
    def launch_dimensions(self):
        return (int(self.params.width), int(self.params.height))


#------------------------------------------------------------------------------
# GLFW callbacks
#------------------------------------------------------------------------------
def mouse_button_callback(window, button, action, mods):
    state = glfw.get_window_user_pointer(window)
    (x, y) = glfw.get_cursor_pos(window)
    if action is glfw.PRESS:
        state.mouse_button = button
        state.trackball.start_tracking(x, y)
    else:
        state.mouse_button = -1

def cursor_position_callback(window, x, y):
    state = glfw.get_window_user_pointer(window)
    if state.mouse_button is glfw.MOUSE_BUTTON_LEFT:
        state.trackball.view_mode = TrackballViewMode.LookAtFixed
        state.trackball.update_tracking(x, y, state.params.width, state.params.height)
        state.camera_changed = True
    elif state.mouse_button is glfw.MOUSE_BUTTON_RIGHT:
        state.trackball.view_mode = TrackballViewMode.EyeFixed
        state.trackball.update_tracking(x, y, state.params.width, state.params.height)
        state.camera_changed = True

def window_size_callback(window, res_x, res_y):
    state = glfw.get_window_user_pointer(window)
    if state.minimized:
        return

    res_x = max(res_x, 1)
    res_y = max(res_y, 1)

    state.params.width = res_x
    state.params.height = res_y
    state.camera_changed = True
    state.resize_dirty = True

def window_iconify_callback(window, iconified):
    state = glfw.get_window_user_pointer(window)
    state.minimized = (iconified > 0)

def key_callback(window, key, scancode, action, mods):
    state = glfw.get_window_user_pointer(window)
    if action == glfw.PRESS:
        if key == glfw.KEY_ESCAPE:
            glfw.set_window_should_close(window, True)
        elif key in {glfw.KEY_ENTER, glfw.KEY_KP_ENTER}:
            state.params.rendering_mode = (int(state.params.rendering_mode) + 1) % len(RenderingMode)
        elif key == glfw.KEY_SPACE:
            state.paused = not state.paused
        elif key == glfw.KEY_UP:
            state.speed *= 1.25
        elif key == glfw.KEY_DOWN:
            state.speed /= 1.25

# letters and symbols go through the char callback, glfw key codes follow the US layout (A is KEY_Q on AZERTY)
def char_callback(window, codepoint):
    state = glfw.get_window_user_pointer(window)
    c = chr(codepoint).lower()
    if c == 'q':
        glfw.set_window_should_close(window, True)
    elif c == 'm':
        state.params.marching_method = (int(state.params.marching_method) + 1) % len(MarchingMethod)
    elif c == 'a':
        state.params.samples_per_pixel = int(state.params.samples_per_pixel) % 8 * 2 or 1
    elif c in {'+', '='}:
        state.speed *= 1.25
    elif c == '-':
        state.speed /= 1.25

def scroll_callback(window, xscroll, yscroll):
    state = glfw.get_window_user_pointer(window)
    if state.trackball.wheel_event(yscroll):
        state.camera_changed = True

#------------------------------------------------------------------------------
# Helper functions
#------------------------------------------------------------------------------
def init_launch_params(state):
    state.params.frame_buffer = 0
    state.params.subframe_index = 0
    state.params.ray_marching_steps = 400
    state.params.ray_marching_threshold = 1e-3
    state.params.rendering_mode = RenderingMode.SHADED
    state.params.time = 0.0
    state.params.marching_method = MarchingMethod.ROOT_WINDOW
    state.params.samples_per_pixel = 2

def handle_camera_update(state):
    if not state.camera_changed:
        return
    state.camera_changed = False

    camera = state.camera
    params = state.params

    camera.aspect_ratio = params.width / float(params.height)
    params.eye = camera.eye

    u,v,w = camera.uvw_frame()
    params.u = u
    params.v = v
    params.w = w

def handle_resize(output_buffer, state):
    if not state.resize_dirty:
        return
    state.resize_dirty = False

    output_buffer.resize(state.params.width, state.params.height)
    alloc_accum_buffer(state)

def alloc_accum_buffer(state):
    state.accum_buffer = cp.empty((int(state.params.height), int(state.params.width), 4), dtype=cp.float32)
    state.params.accum_buffer = state.accum_buffer.data.ptr
    state.params.subframe_index = 0

def update_state(output_buffer, state):
    handle_camera_update(state)
    handle_resize(output_buffer, state)

def launch_subframe(output_buffer, state):
    state.params.frame_buffer = output_buffer.map()

    state.pipeline.launch(state.sbt, dimensions=state.launch_dimensions,
            params=state.params.handle, stream=output_buffer.stream)

    output_buffer.unmap()

def display_subframe(output_buffer, gl_display, window):
    (framebuf_res_x, framebuf_res_y) = glfw.get_framebuffer_size(window)
    gl_display.display( output_buffer.width, output_buffer.height,
                        framebuf_res_x, framebuf_res_y,
                        output_buffer.get_pbo() )

controls = """ENTER  rendering mode
M      marching method
A      samples per pixel
SPACE  pause
+/-    animation speed (or UP/DOWN)
mouse  rotate, zoom
Q/ESC  quit"""

@static_vars(frames=0, last_update_time=None, stats_text="")
def display_overlay(state, state_update_time, render_time, display_time):
    """Same timings as sutil.gui.display_stats, followed by the current settings and the controls."""
    cur_time = glfw.get_time()
    display_overlay.frames += 1
    last_update_time = display_overlay.last_update_time or cur_time - 0.5
    dt = cur_time - last_update_time

    do_update = dt > 0.5 or not display_overlay.stats_text
    if do_update:
        frames = display_overlay.frames
        display_overlay.stats_text = (f"{frames / dt:5.1f} fps\n\n"
            f"state update: {1000.0 * state_update_time / frames:8.1f} ms\n"
            f"render      : {1000.0 * render_time / frames:8.1f} ms\n"
            f"display     : {1000.0 * display_time / frames:8.1f} ms")
        display_overlay.last_update_time = cur_time
        display_overlay.frames = 0

    settings = (f"mode   : {RenderingMode(int(state.params.rendering_mode)).name.lower()}\n"
                f"method : {MarchingMethod(int(state.params.marching_method)).name.lower()}\n"
                f"samples: {int(state.params.samples_per_pixel)} per pixel"
                + (f", {int(state.params.subframe_index) + 1} frames" if state.paused else "") + "\n"
                f"speed  : x{state.speed:.2f}" + (" (paused)" if state.paused else ""))

    imgui.new_frame()
    imgui.set_next_window_bg_alpha(0.0)
    imgui.set_next_window_position(10.0, 10.0)
    imgui.begin("Overlay", None, imgui.WINDOW_NO_TITLE_BAR | imgui.WINDOW_NO_RESIZE | imgui.WINDOW_NO_MOVE |
                imgui.WINDOW_NO_SCROLLBAR | imgui.WINDOW_NO_SAVED_SETTINGS | imgui.WINDOW_NO_INPUTS |
                imgui.WINDOW_ALWAYS_AUTO_RESIZE)
    imgui.push_style_color(imgui.COLOR_TEXT, 0.0, 0.0, 0.0, 1.0)
    imgui.text(f"{display_overlay.stats_text}\n\n{settings}\n\n{controls}")
    imgui.pop_style_color()
    imgui.end()
    imgui.end_frame()

    return do_update


def init_camera_state(state):
    camera = state.camera
    camera.eye = (0, 6.0, -12.5)
    camera.look_at = (0, -0.6, 0)
    camera.up = (0, 1, 0)
    camera.fov_y = 35
    state.camera_changed = True

    trackball = state.trackball
    trackball.move_speed = 10.0
    trackball.set_reference_frame([1,0,0], [0,0,1], [0,1,0])
    trackball.reinitialize_orientation_from_camera()


def create_context(state):
    logger = ox.Logger(cusvna_logger)
    ctx = ox.DeviceContext(validation_mode=validation_mode, log_callback_function=logger, log_callback_level=4)
    ctx.cache_enabled = True
    state.ctx = ctx


def create_module(state):
    pipeline_opts = ox.PipelineCompileOptions(
            uses_motion_blur=False,
            uses_primitive_type_flags=ox.PrimitiveTypeFlags.CUSTOM,
            traversable_graph_flags=ox.TraversableGraphFlags.ALLOW_SINGLE_GAS,
            exception_flags=exception_flags,
            num_payload_values=3,
            num_attribute_values=4,
            pipeline_launch_params_variable_name="params")

    compile_opts = ox.ModuleCompileOptions(
            max_register_count=ox.ModuleCompileOptions.DEFAULT_MAX_REGISTER_COUNT,
            opt_level=opt_level, debug_level=debug_level)

    cuda_source = os.path.join(script_dir, 'cuda', 'levelset.cu')

    example_include_path = os.path.realpath(os.path.join(script_dir, '..'))
    build_flags = without_fast_math(ox.module.get_default_nvrtc_compile_flags(std=17, rdc=True)) + (f'-I{example_include_path}',)

    state.module = ox.Module(state.ctx, cuda_source, compile_opts, pipeline_opts, build_flags)
    state.pipeline_opts = pipeline_opts

def create_program_groups(state):
    ctx, module = state.ctx, state.module

    state.raygen_grp = ox.ProgramGroup.create_raygen(ctx, module, "__raygen__rg")
    state.miss_grps = [ox.ProgramGroup.create_miss(ctx, module, "__miss__radiance"),
                       ox.ProgramGroup.create_miss(ctx, module, "__miss__shadow")]
    state.hit_grp = ox.ProgramGroup.create_hitgroup(ctx, module,
                                              entry_function_IS="__intersection__levelset",
                                              entry_function_CH="__closesthit__ch")

def create_pipeline(state):
    program_grps = [state.raygen_grp, *state.miss_grps, state.hit_grp]

    max_trace_depth = 2  # camera rays and shadow rays
    max_traversable_graph_depth = 1

    link_opts = ox.PipelineLinkOptions(max_trace_depth=max_trace_depth)

    pipeline = ox.Pipeline(state.ctx,
                           compile_options=state.pipeline_opts,
                           link_options=link_opts,
                           program_groups=program_grps,
                           max_traversable_graph_depth=max_traversable_graph_depth)

    pipeline.compute_stack_sizes(max_trace_depth,
                                 0,  # max_cc_depth
                                 0)  # max_dc_depth

    state.pipeline = pipeline

def create_sbt(state):
    raygen_sbt = ox.SbtRecord(state.raygen_grp)

    miss_sbt = ox.SbtRecord(state.miss_grps)

    hit_sbt = ox.SbtRecord(state.hit_grp, names=('objects',), formats=('u8',))
    hit_sbt['objects'] = d_objects.data.ptr

    state.sbt = ox.ShaderBindingTable(raygen_record=raygen_sbt, miss_records=miss_sbt, hitgroup_records=hit_sbt)

def create_gas(state):
    bboxes = cp.asarray(compute_bboxes(objects))
    build_input = ox.BuildInputCustomPrimitiveArray([bboxes], num_sbt_records=1, flags=[ox.GeometryFlags.NONE])
    state.gas = ox.AccelerationStructure(state.ctx, [build_input], compact=True)
    state.params.trav_handle = state.gas.handle

#------------------------------------------------------------------------------
# Main
#------------------------------------------------------------------------------
if __name__ == '__main__':
    cusvna_logger.info("Setting up OptiX scene...")

    state = DynamicGeometryState()
    state.params.width = 1024
    state.params.height = 768
    state.time = 0.0

    buffer_format = BufferImageFormat.UCHAR4
    output_buffer_type = CudaOutputBufferType.enable_gl_interop()

    init_camera_state(state)
    init_launch_params(state)

    create_context(state)
    create_module(state)
    create_program_groups(state)
    create_pipeline(state)
    create_sbt(state)
    create_gas(state)

    window, impl = init_ui("Cusvna Interval Marching", state.params.width, state.params.height)
    glfw.set_mouse_button_callback(window, mouse_button_callback)
    glfw.set_cursor_pos_callback(window, cursor_position_callback)
    glfw.set_window_size_callback(window, window_size_callback)
    glfw.set_window_iconify_callback(window, window_iconify_callback)
    glfw.set_key_callback(window, key_callback)
    glfw.set_char_callback(window, char_callback)
    glfw.set_scroll_callback(window, scroll_callback)
    glfw.set_window_user_pointer(window, state)

    output_buffer = CudaOutputBuffer(output_buffer_type, buffer_format,
            state.params.width, state.params.height)
    alloc_accum_buffer(state)

    gl_display = GLDisplay(buffer_format)

    state_update_time = 0.0
    render_time = 0.0
    display_time = 0.0


    window_title = None
    last_settings = None
    last_time = glfw.get_time()

    while not glfw.window_should_close(window):
        t0 = glfw.get_time()
        glfw.poll_events()

        if not state.paused:
            state.time += (t0 - last_time) * state.speed
        last_time = t0
        state.params.time = state.time

        title = (f"Cusvna Interval Marching - {RenderingMode(int(state.params.rendering_mode)).name.lower()}"
                 f" - {MarchingMethod(int(state.params.marching_method)).name.lower()}"
                 f" - {int(state.params.samples_per_pixel)} spp - speed x{state.speed:.2f}"
                 + (" - paused" if state.paused else ""))
        if title != window_title:
            glfw.set_window_title(window, title)
            window_title = title

        # restart the accumulation of samples whenever the image changes
        settings = (int(state.params.rendering_mode), int(state.params.marching_method), int(state.params.samples_per_pixel))
        if not state.paused or state.camera_changed or state.resize_dirty or settings != last_settings:
            state.params.subframe_index = 0
        last_settings = settings

        update_state(output_buffer, state)

        t1 = glfw.get_time()
        state_update_time += t1 - t0
        t0 = t1

        launch_subframe(output_buffer, state)
        t1 = glfw.get_time()
        render_time += t1 - t0
        t0 = t1

        display_subframe(output_buffer, gl_display, window)
        display_time += t1 - t0

        if display_overlay(state, state_update_time, render_time, display_time):
            state_update_time = 0.0
            render_time = 0.0
            display_time = 0.0

        imgui.render()
        impl.render(imgui.get_draw_data())

        glfw.swap_buffers(window)

        state.params.subframe_index = state.params.subframe_index.item() + 1

    impl.shutdown()
    glfw.terminate()
