import matplotlib.pyplot as plt
import os
import shutil
        
def clear_folder(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.mkdir(path)


def plotPCbatch(pcArray1, pcArray2, show = True, save = False, name=None, fig_count=5 , sizex = 12, sizey=3):
    
    pc1 = pcArray1[:fig_count]
    pc2 = pcArray2[:fig_count]
    
    fig=plt.figure(figsize=(sizex, sizey))
    
    for i in range(fig_count*2):

        ax = fig.add_subplot(2, fig_count,i+1, projection='3d')
        
        if (i<fig_count):
            colors = pc1[i, :, 3:]  #Extract RGB
            ax.scatter(pc1[i,:,0], pc1[i,:,2], pc1[i,:,1], c=colors, marker='.', alpha=0.8, s=8)
        else:
            colors = pc2[i - fig_count, :, 3:]  # Extract RGB
            ax.scatter(pc2[i - fig_count, :, 0], pc2[i - fig_count, :, 2], pc2[i - fig_count, :, 1], c=colors, marker='.', alpha=0.8, s=8)


        ax.set_xlim3d(-0.38, 0.38)
        ax.set_ylim3d(-0.38, 0.38)
        ax.set_zlim3d(-0.38, 0.38)
            
        plt.axis('off')
        
    plt.subplots_adjust(wspace=0, hspace=0)
        
    if(save):
        fig.savefig(name + '.png')
        plt.close(fig)
    
    if(show):
        plt.show()
    else:
        return fig
    
   
 
def plot_and_save_pointcloud(pc, save_path, show=True):
    
    
    """
    Visualizes a single point cloud and saves it as an image.
    
    Args:
        pc: (N, 3) or (N, 6) numpy array representing the point cloud. 
        save_path: Path to save the image.
        show: Whether to display the plot.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    if pc.shape[1] == 6:  # If RGB data is present
        colors = pc[:, 3:]  # Extract RGB
        ax.scatter(pc[:, 0], pc[:, 1], pc[:, 2], c=colors, s=1)
    else:
        ax.scatter(pc[:, 0], pc[:, 1], pc[:, 2], s=1)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.axis('off')

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
        plt.close()
    if show:
        plt.show()
    else:
        plt.close()