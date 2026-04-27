import matplotlib.pyplot as plt
import numpy as np

def plot_propagation(L, all_before, all_after_1, all_after_2, save_svg=False, filename='all_modes.svg'):
    fig = plt.figure(figsize=(10, 8))
    
    extent_val = [-L/2*1e3, L/2*1e3, -L/2*1e3, L/2*1e3]
    mode_names = ['Гауссов пучок', 'ГЛ (0,1)', 'ГЛ (0,2)', 'ГЛ (0,3)', 'ГЛ (0,4)', 'ГЛ (0,5)']
    
    n_rows, n_cols = 6, 6
    
    left_margin = 0.10   
    right_margin = 0.04 
    bottom_margin = 0.06 
    top_margin = 0.08  
    
    grid_width = 1.0 - left_margin - right_margin
    grid_height = 1.0 - bottom_margin - top_margin
    
    frame_width = grid_width / n_cols
    frame_height = grid_height / n_rows
    
    grid_start_x = left_margin
    grid_start_y = bottom_margin
    
    im_for_cbar = {}

    for row in range(n_rows):
        for col in range(n_cols):

            x0 = grid_start_x + col * frame_width
            y0 = grid_start_y + row * frame_height
            
            ax = fig.add_axes([x0, y0, frame_width, frame_height])
            
            if row == 5:
                phase = np.angle(all_before[col])
                im = ax.imshow(phase, cmap='gray', aspect='equal',
                              extent=extent_val, vmin=-np.pi, vmax=np.pi)
                if col == n_cols - 1:
                    im_for_cbar['phase_before'] = (im, x0 + frame_width, y0)
            
            elif row == 4:
                amp = np.abs(all_before[col])
                ax.imshow(amp, cmap='gray_r', aspect='equal', extent=extent_val)
            
            elif row == 3:
                phase = np.angle(all_after_1[col])
                im = ax.imshow(phase, cmap='gray', aspect='equal',
                              extent=extent_val, vmin=-np.pi, vmax=np.pi)
                if col == n_cols - 1:
                    im_for_cbar['phase_r1'] = (im, x0 + frame_width, y0)
            
            elif row == 2:  
                amp = np.abs(all_after_1[col])
                ax.imshow(amp, cmap='gray_r', aspect='equal', extent=extent_val)
            
            elif row == 1: 
                phase = np.angle(all_after_2[col])
                im = ax.imshow(phase, cmap='gray', aspect='equal',
                              extent=extent_val, vmin=-np.pi, vmax=np.pi)
                if col == n_cols - 1:
                    im_for_cbar['phase_r2'] = (im, x0 + frame_width, y0)
            
            elif row == 0: 
                amp = np.abs(all_after_2[col])
                ax.imshow(amp, cmap='gray_r', aspect='equal', extent=extent_val)
            
            ax.axis('off')
    

    for key, (im, x_pos, y_pos) in im_for_cbar.items():
        cbar_width = 0.01  
        cbar_x = x_pos + 0.005  
        cbar_ax = fig.add_axes([cbar_x, y_pos, cbar_width, frame_height])
        cbar = fig.colorbar(im, cax=cbar_ax)
        cbar.set_ticks([-np.pi + 0.2, 0, np.pi - 0.2])
        cbar.set_ticklabels(['-π', '0', 'π'])
        cbar.ax.tick_params(labelsize=11) 
    
    row_labels = ['Ампл. \n(r0=0.02)', 'Фаза \n(r0=0.02)',
                  'Ампл. \n(r0=0.01)', 'Фаза \n(r0=0.01)',
                  'Ампл. до', 'Фаза до']
    
    for row in range(n_rows):
        text_x = left_margin - 0.008  
        text_y = grid_start_y + row * frame_height + frame_height/2
        fig.text(text_x, text_y, row_labels[row], 
                fontsize=10, va='center', ha='right', 
                linespacing=1.2)  
    
    for col in range(n_cols):
        text_x = grid_start_x + col * frame_width + frame_width/2
        text_y = grid_start_y + n_rows * frame_height + 0.005 
        fig.text(text_x, text_y, mode_names[col], 
                fontsize=11, va='bottom', ha='center')  
    
    if save_svg:
        plt.savefig(filename, format='svg', dpi=300, bbox_inches='tight', pad_inches=0)
        print(f"График сохранен как {filename}")
    
    plt.show()