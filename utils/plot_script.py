import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation, PillowWriter
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import mpl_toolkits.mplot3d.axes3d as p3
from textwrap import wrap

def plot_3d_pose(kinematic_tree, joints, title, figsize=(10, 10), radius=4):
    
    title = "\n".join(wrap(title, 50))

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim3d([-radius / 2, radius / 2])
    ax.set_ylim3d([0, 2*radius])
    ax.set_zlim3d([0, radius])

    fig.suptitle(title, fontsize=20)
    ax.grid(b=False)

    def plot_xzPlane(minx, maxx, miny, minz, maxz):
        ## Plot a plane XZ
        verts = [
            [minx, miny, minz],
            [minx, miny, maxz],
            [maxx, miny, maxz],
            [maxx, miny, minz]
        ]
        xz_plane = Poly3DCollection([verts])
        xz_plane.set_facecolor((0.5, 0.5, 0.5, 0.5))
        ax.add_collection3d(xz_plane)
        
    data = joints.copy()
    MINS = data.min(axis=0)
    MAXS = data.max(axis=0)
    colors = ['#A569BD', '#A569BD', '#A569BD', '#A569BD', '#A569BD', '#A569BD',
              '#26A69A', '#26A69A', '#26A69A', '#26A69A', '#26A69A',
             '#FFCA28', '#FFCA28','#FFCA28','#FFCA28','#FFCA28',]

    height_offset = MINS[1]
    data[:, 1] -= height_offset
    
    data[..., 0] -= data[0:1, 0]
    data[..., 2] -= data[0:1, 2]
   
    ax.view_init(elev=120, azim=-90)
        
    # plot_xzPlane(MINS[0], MAXS[0], -1, MINS[2], MAXS[2])
    for i, (chain, color) in enumerate(zip(kinematic_tree, colors)):
        if i < 6:
            linewidth = 4.0
        else:
            linewidth = 2.0
        ax.plot3D(data[chain, 0], data[chain, 1], data[chain, 2], linewidth=linewidth, color=color)
        
    plt.axis('off')
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])


def plot_3d_motion(save_path, kinematic_tree, joints, title, figsize=(10, 10), fps=120, radius=4):
#     matplotlib.use('Agg')
    
    title = "\n".join(wrap(title, 60))

    def init():
        ax.set_xlim3d([-radius / 2, radius / 2])
        ax.set_ylim3d([0, 2*radius])
        ax.set_zlim3d([0, radius])
        # print(title)
        fig.suptitle(title, fontsize=20)
        ax.grid(b=False)

    def plot_xzPlane(minx, maxx, miny, minz, maxz):
        ## Plot a plane XZ
        verts = [
            [minx, miny, minz],
            [minx, miny, maxz],
            [maxx, miny, maxz],
            [maxx, miny, minz]
        ]
        xz_plane = Poly3DCollection([verts])
        xz_plane.set_facecolor((0.1, 0.1, 0.1, 0.1))
        ax.add_collection3d(xz_plane)

    #         return ax

    # (seq_len, joints_num, 3)
    data = joints.copy().reshape(len(joints), -1, 3)
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    init()
    MINS = data.min(axis=0).min(axis=0)
    MAXS = data.max(axis=0).max(axis=0)
    colors = ['#A569BD', '#A569BD', '#A569BD', '#A569BD', '#A569BD', '#A569BD',  
              '#26A69A', '#26A69A', '#26A69A', '#26A69A', '#26A69A',
             '#FFCA28', '#FFCA28','#FFCA28','#FFCA28','#FFCA28',]
    frame_number = data.shape[0]
    #     print(data.shape)

    height_offset = MINS[1]
    data[:, :, 1] -= height_offset
    # left_trajec = data[:, 24, [0, 2]]
    # right_trajec = data[:, 39, [0, 2]]
    
    # data[..., 0] -= data[:, 0:1, 0]
    # data[..., 2] -= data[:, 0:1, 2]

    #     print(trajec.shape)

    def update(index):
        #         print(index)
        # ax.lines = []
        # ax.collections = []
        # Clear existing lines and collections
        for line in ax.lines:
            line.remove()
        for collection in ax.collections:
            collection.remove()
        ax.view_init(elev=120, azim=-90)
        
        plot_xzPlane(MINS[0], MAXS[0], -1, MINS[2], MAXS[2])
        
        # if index > 1:
        #     ax.plot3D(left_trajec[:index, 0]-left_trajec[index, 0], np.zeros_like(left_trajec[:index, 0]), left_trajec[:index, 1]-left_trajec[index, 1], linewidth=1.0,
        #               color='#26A69A')
        #     ax.plot3D(right_trajec[:index, 0]-right_trajec[index, 0], np.zeros_like(right_trajec[:index, 0]), right_trajec[:index, 1]-right_trajec[index, 1], linewidth=1.0,
        #               color='#FFCA28')
        
        for i, (chain, color) in enumerate(zip(kinematic_tree, colors)):
            if i < 6:
                linewidth = 4.0
            else:
                linewidth = 2.0
            ax.plot3D(data[index, chain, 0], data[index, chain, 1], data[index, chain, 2], linewidth=linewidth, color=color)
        
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])

    ani = FuncAnimation(fig, update, frames=frame_number, interval=1000/fps, repeat=False)

    ani.save(save_path, fps=fps)
    plt.close()
    
    print('done !')