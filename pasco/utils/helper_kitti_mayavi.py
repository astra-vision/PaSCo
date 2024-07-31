import numpy as np
from mayavi import mlab
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import seaborn as sns
from matplotlib.colors import ListedColormap
try:
    engine = mayavi.engine
except NameError:
    from mayavi.api import Engine
    engine = Engine()
    engine.start()
    

kitti_ssc_colors = np.array([
        [0  , 0  , 0, 255],
        [100, 150, 245, 255],
        [100, 230, 245, 255],
        [30, 60, 150, 255],
        [80, 30, 180, 255],
        [100, 80, 250, 255],
        [255, 30, 30, 255],
        [255, 40, 200, 255],
        [150, 30, 90, 255],
        [255, 0, 255, 255],
        [255, 150, 255, 255],
        [75, 0, 75, 255],
        [175, 0, 75, 255],
        [255, 200, 0, 255],
        [255, 120, 50, 255],
        [0, 175, 0, 255],
        [135, 60, 0, 255],
        [150, 240, 80, 255],
        [255, 240, 150, 255],
        [255, 0, 0, 255]]).astype(np.uint8)


kitti360_ssc_colors = np.array([
        [0  , 0  , 0, 255],
        [100, 150, 245, 255], 
        [100, 230, 245, 255], 
        [30, 60, 150, 255], 
        [80, 30, 180, 255], 
        [0, 0, 255, 255], 
        [255, 30, 30, 255], 
        [255, 0, 255, 255], 
        [255, 150, 255, 255], 
        [75, 0, 75, 255], 
        [175, 0, 75, 255], 
        [255, 200, 0, 255], 
        [255, 120, 50, 255], 
        [0, 175, 0, 255], 
        [150, 240, 80, 255], 
        [255, 240, 150, 255], 
        [255, 0, 0, 255], 
        [255, 150, 0, 255], 
        [50, 255, 255, 255]])





def position_scene_view(scene, view=1):
  
    if view == 1:
        scene.x_minus_view()
        scene.camera.position = [-94.41310048749838, 22.126089091388785, 74.40160208248852]
        scene.camera.focal_point = [25.49999923631549, 25.49999923631549, 1.9999999515712261]
        scene.camera.view_angle = 30.0
        scene.camera.view_up = [0.5167491230431891, 0.005722545424782316, 0.856117746754473]
        scene.camera.clipping_range = [91.73035948285289, 201.3617893713614]
        scene.camera.compute_view_plane_normal()
        scene.render()
        scene.camera.position = [-51.927412713828446, -46.660344208409334, 93.81725028508137]
        scene.camera.focal_point = [25.49999923631549, 25.49999923631549, 1.9999999515712261]
        scene.camera.view_angle = 30.0
        scene.camera.view_up = [0.5318969793651769, 0.38739945081093075, 0.7529988504995144]
        scene.camera.clipping_range = [81.58874876428513, 214.12798778846422]
        scene.camera.compute_view_plane_normal()
        scene.render()
        scene.camera.position = [-54.665532379571125, -43.712070618513835, 93.7371444096225]
        scene.camera.focal_point = [25.49999923631549, 25.49999923631549, 1.9999999515712261]
        scene.camera.view_angle = 30.0
        scene.camera.view_up = [0.5442689178381115, 0.3685077360527552, 0.7536400954995723]
        scene.camera.clipping_range = [81.66754657666122, 214.0287975774116]
        scene.camera.compute_view_plane_normal()
        scene.render()
    elif view == 2:
        scene.x_minus_view()
        scene.camera.position = [-73.50517293453248, 20.31870450814589, 86.55895209276665]
        scene.camera.focal_point = [25.30000038072467, 24.899999976158142, 3.1000000946223736]
        scene.camera.view_angle = 30.0
        scene.camera.view_up = [0.6443675997739866, 0.02593553627145462, 0.764275960841255]
        scene.camera.clipping_range = [83.90501278412131, 187.0201395230652]
        scene.camera.compute_view_plane_normal()
        scene.render()
        scene.camera.position = [-57.76202659949473, 12.831974999732642, 101.60842439305252]
        scene.camera.focal_point = [25.30000038072467, 24.899999976158142, 3.1000000946223736]
        scene.camera.view_angle = 30.0
        scene.camera.view_up = [0.7475253857045429, 0.14540009359025238, 0.6481239160154639]
        scene.camera.clipping_range = [86.82663418580873, 183.34242012045678]
        scene.camera.compute_view_plane_normal()
        scene.render()
        scene.camera.position = [-63.19573974005762, -16.24370662463224, 88.09733298382173]
        scene.camera.focal_point = [25.30000038072467, 24.899999976158142, 3.1000000946223736]
        scene.camera.view_angle = 30.0
        scene.camera.view_up = [0.622797136058541, 0.2146867446205395, 0.7523518651545396]
        scene.camera.clipping_range = [75.71218101079536, 197.33322675533944]
        scene.camera.compute_view_plane_normal()
        scene.render()
    elif view == 3:
        scene.x_minus_view()
        scene.camera.position = [-87.32520967897096, 88.15719174181817, 11.033281860943813]
        scene.camera.focal_point = [25.30000038072467, 24.899999976158142, 3.1000000946223736]
        scene.camera.view_angle = 30.0
        scene.camera.view_up = [0.053234459204747446, -0.030396851119650972, 0.998119293368878]
        scene.camera.clipping_range = [62.73532338749172, 213.66841687912512]
        scene.camera.compute_view_plane_normal()
        scene.render()
        scene.camera.position = [-46.89647976374451, 85.86325167115542, 91.53074931780868]
        scene.camera.focal_point = [25.30000038072467, 24.899999976158142, 3.1000000946223736]
        scene.camera.view_angle = 30.0
        scene.camera.view_up = [0.5742636948802018, -0.375309619863306, 0.727573981104739]
        scene.camera.clipping_range = [75.32641418079382, 197.81882771469628]
        scene.camera.compute_view_plane_normal()
        scene.render()

    return

    
  
    return

def get_grid_coords(dims, resolution):
  '''
  :param dims: the dimensions of the grid [x, y, z] (i.e. [256, 256, 32])
  :return coords_grid: is the center coords of voxels in the grid
  '''

  # The sensor in centered in X (we go to dims/2 + 1 for the histogramdd)
  g_xx = np.arange(0, dims[0] + 1)
  # The sensor is in Y=0 (we go to dims + 1 for the histogramdd)
  g_yy = np.arange(0, dims[1] + 1)
  # The sensor is in Z=1.73. I observed that the ground was to voxel levels above the grid bottom, so Z pose is at 10
  # if bottom voxel is 0. If we want the sensor to be at (0, 0, 0), then the bottom in z is -10, top is 22
  # (we go to 22 + 1 for the histogramdd)
  # ATTENTION.. Is 11 for old grids.. 10 for new grids (v1.1) (https://github.com/PRBonn/semantic-kitti-api/issues/49)
  sensor_pose = 10
  g_zz = np.arange(0, dims[2] + 1)

  # Obtaining the grid with coords...
  xx, yy, zz = np.meshgrid(g_xx[:-1], g_yy[:-1], g_zz[:-1])

  coords_grid = np.array([xx.flatten(), yy.flatten(), zz.flatten()]).T
  coords_grid = coords_grid.astype(np.float)

  coords_grid = (coords_grid * resolution) + resolution/2

  temp = np.copy(coords_grid)
  temp[:, 0] = coords_grid[:, 1]
  temp[:, 1] = coords_grid[:, 0]
  coords_grid = np.copy(temp)

  return coords_grid, g_xx, g_yy, g_zz

def draw_tsdf(scans, voxel_size=0.08):
    figure = mlab.figure(size = (1400,1400),\
                    bgcolor = (1,1,1))
    t = 0
    for voxels in scans:
        t += 1
        grid_coords, _, _, _ = get_grid_coords([voxels.shape[0], voxels.shape[2], voxels.shape[1]], voxel_size)    
        grid_coords = np.vstack((grid_coords.T, np.moveaxis(voxels, [0, 1, 2], [0, 2, 1]).reshape(-1))).T
    
        # Obtaining voxels with semantic class
        # occupied_voxels = grid_coords[((grid_coords[:, 3]) < 0.0) & ((grid_coords[:, 3]) > -10) ]
        # occupied_voxels = grid_coords[((grid_coords[:, 3]) > 0)  ]
        occupied_voxels = grid_coords[((grid_coords[:, 3]) == 1)  ]
        # occupied_voxels = grid_coords
        opa = 1.0
        if t == 2:
            opa = 1.0

        plt_plot = mlab.points3d(
            occupied_voxels[:, 0], occupied_voxels[:, 1], occupied_voxels[:, 2], 
            occupied_voxels[:, 3],
            colormap='magma', scale_factor=0.075,
            mode='cube', opacity=opa,
            vmin=-1, vmax=0)

        plt_plot.glyph.scale_mode = 'scale_by_vector'

    # mlab.show()
    mlab.savefig(out_filepath, figure=figure)
    mlab.clf()


def draw_coords(
        coords,
        voxel_size=0.2,
        view=1):
    # grid_coords, _, _, _ = get_grid_coords([voxels.shape[0], voxels.shape[1], voxels.shape[2]], voxel_size)

    # print(grid_coords.shape)
    # coords = np.vstack([grid_coords.T, voxels.reshape(-1)]).T
    # print(grid_coords.shape, valid_pix.shape)
    # fov_grid_coords = grid_coords[valid_pix, :]
    # outfov_grid_coords = grid_coords[~valid_pix, :]

    # Obtaining voxels with semantic class
    # voxels = grid_coords[(grid_coords[:, 3] > 0) & (grid_coords[:, 3] < 255)]
    # outfov_voxels = outfov_grid_coords[(outfov_grid_coords[:, 3] > 0) & (outfov_grid_coords[:, 3] < 255)]

    figure = mlab.figure(size=(1400, 1400), bgcolor=(1, 1, 1), engine=engine)

    plt_plot = mlab.points3d(coords[:, 0], coords[:, 1], coords[:, 2], scale_factor=voxel_size - 0.05 * voxel_size, mode='cube',
                            color=(0.8, 0.8, 0.8),
                             opacity=1.)

    position_scene_view(figure.scene, view)
    colors = np.array([
        # [0  , 0  , 0, 255],
        [100, 150, 245, 255],
        [100, 230, 245, 255],
        [30, 60, 150, 255],
        [80, 30, 180, 255],
        [100, 80, 250, 255],
        [255, 30, 30, 255],
        [255, 40, 200, 255],
        [150, 30, 90, 255],
        [255, 0, 255, 255],
        [255, 150, 255, 255],
        [75, 0, 75, 255],
        [175, 0, 75, 255],
        [255, 200, 0, 255],
        [255, 120, 50, 255],
        [0, 175, 0, 255],
        [135, 60, 0, 255],
        [150, 240, 80, 255],
        [255, 240, 150, 255],
        [255, 0, 0, 255]]).astype(np.uint8)
    
    

    plt_plot.glyph.scale_mode = 'scale_by_vector'

    plt_plot.module_manager.scalar_lut_manager.lut.table = colors

    mlab.show()



def draw_semantic(
        coords, labels,
        filename=None,
        voxel_size=0.2,
        figure=None,
        view=1):


    mask = (labels != 0) & (labels != 255)
    coords = coords[mask]
    labels = labels[mask]
    if figure is None:
        figure = mlab.figure(size=(1400, 1400), bgcolor=(1, 1, 1), engine=engine)
    points = coords * voxel_size
    
    plt_plot = mlab.points3d(points[:, 0], points[:, 1], points[:, 2], labels,
                            #  colormap='gist_ncar', 
                            #  scale_factor=0.95,
                            #  scale_factor=voxel_size - 0.05 * voxel_size, 
                             scale_factor=voxel_size,
                             figure=figure,
                             mode='cube',
                             vmin=1, vmax=19)
    

    position_scene_view(figure.scene, view)
    colors = np.array([
        # [0  , 0  , 0, 255],
        [100, 150, 245, 255],
        [100, 230, 245, 255],
        [30, 60, 150, 255],
        [80, 30, 180, 255],
        [100, 80, 250, 255],
        [255, 30, 30, 255],
        [255, 40, 200, 255],
        [150, 30, 90, 255],
        [255, 0, 255, 255],
        [255, 150, 255, 255],
        [75, 0, 75, 255],
        [175, 0, 75, 255],
        [255, 200, 0, 255],
        [255, 120, 50, 255],
        [0, 175, 0, 255],
        [135, 60, 0, 255],
        [150, 240, 80, 255],
        [255, 240, 150, 255],
        [255, 0, 0, 255]]).astype(np.uint8)

    # colors = np.array([
    #     # [0  , 0  , 0, 255],
    #     [100, 150, 245, 255],
        
    #     [30, 60, 150, 255],
    #     [100, 230, 245, 255],
        
    #     [80, 30, 180, 255],
    #     [100, 80, 250, 255],
    #     [255, 30, 30, 255],
    #     [255, 40, 200, 255],
    #     [150, 30, 90, 255],
    #     [255, 0, 255, 255],
    #     [255, 150, 255, 255],
    #     [75, 0, 75, 255],
    #     [175, 0, 75, 255],
    #     [255, 200, 0, 255],
    #     [255, 120, 50, 255],
    #     [0, 175, 0, 255],
    #     [135, 60, 0, 255],
    #     [150, 240, 80, 255],
    #     [255, 240, 150, 255],
    #     [255, 0, 0, 255]]).astype(np.uint8)

    plt_plot.glyph.scale_mode = 'scale_by_vector'
    plt_plot.module_manager.scalar_lut_manager.lut.table = colors

    if filename is not None:
        mlab.savefig(filename, figure=figure)
        mlab.clf()
    else:
        mlab.show()
        

def draw_instance(
    thing_coords, thing_labels,
    voxel_size=0.2, 
    figure=None, filename=None,
    view=1):    
    # colors1 = sns.color_palette("hls", 20)
    colors2 = sns.color_palette("tab20c")
    # colors3 = sns.color_palette("tab20b")
    # colors4 = sns.color_palette("tab20")
    colors5 = sns.color_palette("Set1")
    colors6 = sns.color_palette("Set2")
    colors7 = sns.color_palette("Set3")
    # colors8 = sns.color_palette("Accent")
    colors = list(colors7) + list(colors6)
    colors = [[*[k*255 for k in t], 255] for t in colors]
    max_instances = max(thing_labels)
    if figure is None:
        figure = mlab.figure(size = (1400,1400), bgcolor = (1,1,1), engine=engine)

    # mask = (stuff_labels != 0) & (stuff_labels != 255)
    # stuff_coords = stuff_coords[mask]
    # stuff_labels = stuff_labels[mask]   
    # stuff_points = stuff_coords * voxel_size
    # ssc_plot= mlab.points3d(stuff_points[:, 0], stuff_points[:, 1], stuff_points[:, 2], 
    #                         stuff_labels,
    #                         # colormap='afmhot', 
    #                         # scale_factor=voxel_size - 0.05 * voxel_size, 
    #                         scale_factor=voxel_size,
    #                         mode='cube', 
    #                         figure=figure,
    #                         vmin=0, vmax=19,
    #                         opacity=1.0)


    mask = (thing_labels != 0) & (thing_labels != 255)
    thing_coords = thing_coords[mask]
    thing_labels = thing_labels[mask]   
    thing_points = thing_coords * voxel_size

    
    # points = coords * voxel_size
    thing_plot= mlab.points3d(thing_points[:, 0], thing_points[:, 1], thing_points[:, 2], thing_labels,
                            # colormap='gnuplot2', 
                            scale_factor=0.95 * voxel_size, mode='cube', 
                            vmin=1, vmax=max_instances,
                            figure=figure,
                            opacity=1.)

    position_scene_view(figure.scene, view)
    
    # ssc_plot.glyph.scale_mode = 'scale_by_vector'
    thing_plot.glyph.scale_mode = 'scale_by_vector'


    thing_plot.module_manager.scalar_lut_manager.lut.table = colors
    
    # Making the colors darker
    # A common approach is to reduce the brightness by a certain percentage
    darkening_factor = 0.3  # Reduce brightness by 30%
    # darker_colors = np.clip(ssc_colors[:, :3] * darkening_factor, 0, 255).astype(np.uint8)

    # Keeping the alpha channel unchanged
    # darker_colors_with_alpha = np.column_stack((darker_colors, ssc_colors[:, 3]))


    # ssc_plot.module_manager.scalar_lut_manager.lut.table = darker_colors_with_alpha

    if filename is not None:
        mlab.savefig(filename, figure=figure)
        mlab.clf()
    else:
        mlab.show()
    
   
def draw_panoptic(
    thing_coords, thing_labels,
    stuff_coords, stuff_labels,
    voxel_size=0.2, 
    figure=None,
    # max_instances=100,
    dataset="kitti",
    filename=None,
    view=1):    
    
    
    
    # colors1 = sns.color_palette("hls", 20)
    # colors2 = sns.color_palette("tab20c")
    # colors3 = sns.color_palette("tab20b")
    # colors4 = sns.color_palette("tab20")
    # colors5 = sns.color_palette("Set1")
    # colors6 = sns.color_palette("Set2")
    colors7 = sns.color_palette("Set3")
    
    colors8 = sns.color_palette("Accent")
    # colors = list(colors1) + list(colors2) + list(colors3) + list(colors4) + list(colors5) + list(colors6) + list(colors7) + list(colors8)
    # colors =  list(colors2) + list(colors3) + list(colors4) + list(colors5) + list(colors6) + list(colors7) # + list(colors8)
    colors = list(colors7) + list(colors8)
    print("len colors", len(colors))
    colors = [[*[k*255 for k in t], 255] for t in colors]
    
    if figure is None:
        figure = mlab.figure(size = (1400,1400), bgcolor = (1,1,1), engine=engine)

    mask = (stuff_labels != 0) & (stuff_labels != 255)
    stuff_coords = stuff_coords[mask]
    stuff_labels = stuff_labels[mask]   
    stuff_points = stuff_coords * voxel_size
    
    if dataset == "kitti":
        vmax = 19
    elif dataset == "kitti360":
        vmax = 18
    ssc_plot= mlab.points3d(stuff_points[:, 0], stuff_points[:, 1], stuff_points[:, 2], 
                            stuff_labels,
                            # colormap='afmhot', 
                            # scale_factor=voxel_size - 0.05 * voxel_size, 
                            scale_factor=voxel_size,
                            mode='cube', 
                            figure=figure,
                            vmin=0, vmax=vmax,
                            opacity=1.0)

    if len(thing_labels) > 0:
        mask = (thing_labels != 0) & (thing_labels != 255)
        thing_coords = thing_coords[mask]
        thing_labels = thing_labels[mask]   
        thing_points = thing_coords * voxel_size
        
        # points = coords * voxel_size
        thing_plot= mlab.points3d(thing_points[:, 0], thing_points[:, 1], thing_points[:, 2], thing_labels,
                                # colormap='gnuplot2', 
                                scale_factor=0.95 * voxel_size, mode='cube', 
                                vmin=1, vmax=len(colors),
                                figure=figure,
                                opacity=1.)
        thing_plot.glyph.scale_mode = 'scale_by_vector'


        thing_plot.module_manager.scalar_lut_manager.lut.table = colors

    position_scene_view(figure.scene, view)
    
    ssc_plot.glyph.scale_mode = 'scale_by_vector'
    
    if dataset == "kitti":
        ssc_colors = kitti_ssc_colors
    else:
        ssc_colors = kitti360_ssc_colors
    ssc_plot.module_manager.scalar_lut_manager.lut.table = ssc_colors

    if filename is not None:
        mlab.savefig(filename, figure=figure)
        mlab.clf()
    else:
        mlab.show()
   
    
def draw_panoptic(
    thing_coords, thing_labels,
    stuff_coords, stuff_labels,
    voxel_size=0.2, 
    figure=None,
    # max_instances=100,
    dataset="kitti",
    filename=None,
    view=1):    
    
    
    
    # colors1 = sns.color_palette("hls", 20)
    # colors2 = sns.color_palette("tab20c")
    # colors3 = sns.color_palette("tab20b")
    # colors4 = sns.color_palette("tab20")
    # colors5 = sns.color_palette("Set1")
    # colors6 = sns.color_palette("Set2")
    colors7 = sns.color_palette("Set3")
    
    colors8 = sns.color_palette("Accent")
    # colors = list(colors1) + list(colors2) + list(colors3) + list(colors4) + list(colors5) + list(colors6) + list(colors7) + list(colors8)
    # colors =  list(colors2) + list(colors3) + list(colors4) + list(colors5) + list(colors6) + list(colors7) # + list(colors8)
    colors = list(colors7) + list(colors8)
    print("len colors", len(colors))
    colors = [[*[k*255 for k in t], 255] for t in colors]
    
    if figure is None:
        figure = mlab.figure(size = (1400,1400), bgcolor = (1,1,1), engine=engine)

    mask = (stuff_labels != 0) & (stuff_labels != 255)
    stuff_coords = stuff_coords[mask]
    stuff_labels = stuff_labels[mask]   
    stuff_points = stuff_coords * voxel_size
    
    if dataset == "kitti":
        vmax = 19
    elif dataset == "kitti360":
        vmax = 18
    ssc_plot= mlab.points3d(stuff_points[:, 0], stuff_points[:, 1], stuff_points[:, 2], 
                            stuff_labels,
                            # colormap='afmhot', 
                            # scale_factor=voxel_size - 0.05 * voxel_size, 
                            scale_factor=voxel_size,
                            mode='cube', 
                            figure=figure,
                            vmin=0, vmax=vmax,
                            opacity=1.0)

    if len(thing_labels) > 0:
        mask = (thing_labels != 0) & (thing_labels != 255)
        thing_coords = thing_coords[mask]
        thing_labels = thing_labels[mask]   
        thing_points = thing_coords * voxel_size
        
        # points = coords * voxel_size
        thing_plot= mlab.points3d(thing_points[:, 0], thing_points[:, 1], thing_points[:, 2], thing_labels,
                                # colormap='gnuplot2', 
                                scale_factor=0.95 * voxel_size, mode='cube', 
                                vmin=1, vmax=len(colors),
                                figure=figure,
                                opacity=1.)
        thing_plot.glyph.scale_mode = 'scale_by_vector'


        thing_plot.module_manager.scalar_lut_manager.lut.table = colors

    position_scene_view(figure.scene, view)
    
    ssc_plot.glyph.scale_mode = 'scale_by_vector'
    
    if dataset == "kitti":
        ssc_colors = kitti_ssc_colors
    else:
        ssc_colors = kitti360_ssc_colors
    ssc_plot.module_manager.scalar_lut_manager.lut.table = ssc_colors

    if filename is not None:
        mlab.savefig(filename, figure=figure)
        mlab.clf()
    else:
        mlab.show()
    


    

def draw_uncertainty(
    coords, labels,
    voxel_size=0.2,
    view=1,
    vrange=(0, 1),
    filename=None,
    figure=None):    
    # from tvtk.api import tvtk
    mask = (labels != 0) & (labels != 255) # remove empty voxel and unknown class
    coords = coords[mask]
    labels = labels[mask]
    # coords = coords[labels != 255]
    # labels = labels[labels != 255]
    
    points = coords * voxel_size
    # labels = 1 - labels
    # colormap = cm.inferno
    # vmin=np.min(labels) if len(labels) > 0 else 0
    # vmax=np.max(labels) if len(labels) > 0 else 1
    vmin = vrange[0]
    vmax = vrange[1]
    # normalize = mcolors.Normalize(vmin=vmin, vmax=vmax)
    # s_map = cm.ScalarMappable(norm=normalize, cmap=colormap)
    # colors = s_map.to_rgba(labels) * 255
   
    if figure is None:
        figure = mlab.figure(size = (1400,1400), bgcolor = (1,1,1), engine=engine)
    
    plt_plot= mlab.points3d(points[:, 0], points[:, 1], points[:, 2], labels,
                            colormap='RdYlGn', 
                            scale_factor=voxel_size - 0.05 * voxel_size, mode='cube', 
                            vmin=vmin, vmax=vmax,
                            figure=figure,
                            opacity=1.0)

    position_scene_view(figure.scene, view)
    
    plt_plot.glyph.scale_mode = 'scale_by_vector'
    # plt_plot.module_manager.scalar_lut_manager.lut.table = colors

    if filename is None:
        mlab.show()
    else:
        mlab.savefig(filename, figure=figure)
        mlab.clf()

    
    
def draw_pcd(
    points,
    size=0.2,
    figure=None,
    filename=None,
    view=1):    
    # from tvtk.api import tvtk
    # mask = (labels != 0) & (labels != 255) # remove empty voxel and unknown class
    # coords = coords[mask]
    # labels = labels[mask]
    # coords = coords[labels != 255]
    # labels = labels[labels != 255]
    
    # points = coords * voxel_size
    # labels = 1 - labels
    # colormap = cm.inferno
    # vmin=np.min(labels)
    # vmax=np.max(labels)
    # normalize = mcolors.Normalize(vmin=vmin, vmax=vmax)
    # s_map = cm.ScalarMappable(norm=normalize, cmap=colormap)
    # colors = s_map.to_rgba(labels) * 255
   
    if figure is None:
        figure = mlab.figure(size = (1400,1400), bgcolor = (1,1,1), engine=engine)
    
    plt_plot= mlab.points3d(points[:, 0], points[:, 1], points[:, 2],
                            color=(0.8, 0.8, 0.8), 
                            scale_factor=size, mode='sphere', 
                            # vmin=vmin, vmax=vmax,
                            figure=figure,
                            opacity=1.0)

    position_scene_view(figure.scene, view)
    
    plt_plot.glyph.scale_mode = 'scale_by_vector'
    # plt_plot.module_manager.scalar_lut_manager.lut.table = colors

    if filename is None:
        mlab.show()
    else:
        mlab.savefig(filename, figure=figure)
        mlab.clf()

