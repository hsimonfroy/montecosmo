import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from ipywidgets import interactive, fixed, IntSlider, FloatSlider
from os.path import join
from pathlib import Path
from IPython.display import display
# Accesses global variables dict of __main__, see also vars(__main__). 
# globals() call would provide global variables of this module scope only.
from __main__ import __dict__





########################
# Save plots interface #
########################
def save_current_subplots(fig_directory, fig_name):

    # Assert fig_name does not include further directories path or other improper chars
    forbidden_chars = ["/", "\\", ":", "\"", "?"] # NOTE: backslash is "\\" or chr(92)
    assert all([forbidden_char not in fig_name for forbidden_char in forbidden_chars]), f"output_name should not contain these chars: {forbidden_chars}"
    
    # Set output directory 
    Path(fig_directory).mkdir(parents=True, exist_ok=True)

    # NOTE: if an extension is provided in the name, it is selected and appended to subplots names
    # e.g. "FIG.jpg" would save "FIG.jpg" figure and "FIG_ax1.jpg", "FIG_ax2.jpg" subplots
    fig_name_stem_suffix = Path(fig_name)
    fig_name_stem, fig_name_suffix = fig_name_stem_suffix.stem, fig_name_stem_suffix.suffix

    # Save figure and its subplots
    figure_fullname = join(fig_directory, fig_name)
    plt.savefig(figure_fullname)
    axes_list = plt.gcf().axes
    print("Saving...", end="\r")
    if axes_list:
        for i_ax, ax in enumerate(axes_list):
            extent = ax.get_tightbbox().transformed(plt.gcf().dpi_scale_trans.inverted()).padded(0.1/2.54) # add 0.1 cm of padding (matplotlib unit is inches)
            subplot_fullname = join(fig_directory, fig_name_stem + "_ax" + str(i_ax+1) + fig_name_suffix)
            plt.savefig(subplot_fullname, bbox_inches=extent)
        print(f"Saved figure {fig_name} and its {i_ax+1} subplots in {fig_directory} folder.")
    else:
        print(f"Saved figure {fig_name} in {fig_directory} folder.")    


def set_plotting_options(use_TeX, font_size):

    # Reset plotting options, in case they has been coincidentaly altered
    mpl.rcParams.update(mpl.rcParamsDefault)
    
    # Set plotting options
    params = {'axes.titlesize': font_size,
              'axes.labelsize': font_size,
              'xtick.labelsize': font_size,
              'ytick.labelsize': font_size,
              'legend.fontsize': font_size,
              'text.usetex': use_TeX,
              'ps.useafm': True,
              'pdf.use14corefonts': True} # NOTE: 'ps.useafm' and 'pdf.use14corefonts' for PS and PDF font comptatibiliies
    plt.rcParams.update(params)


def plot_figure(my_plot, fig_width=6.4, fig_height=4.8, use_TeX=False, font_size=10):
    
    # Default plotting parameters are matplotlib default options
    set_plotting_options(use_TeX, font_size)

    # Close potential previous figures to not saturate cache
    plt.close()

    # Create and display figure
    plt.figure(figsize=(fig_width, fig_height))
    my_plot()
    plt.tight_layout()
    display(plt.gcf()) 
    # NOTE: use display() instead of plt.show(), because the latter close figure automatically which doesn't allow to save it later. 
    # However, it is better not to forget to close figure later.


def get_save_plots_interface(my_plot, 
                    fig_size=(6.4, 4.8), use_TeX=False, font_size=10,
                    fig_directory="./figs/", fig_name="FIG"):
    # NOTE: default plotting parameters are matplotlib default options

    # Construct plot figure button
    fig_width, fig_height = fig_size
    plot_fig_button = interactive(plot_figure, {"manual":True, "manual_name":"plot figure"},
                              my_plot=fixed(my_plot),
                              fig_width=FloatSlider(min=4., max=20., step=.2, value=fig_width, continuous_update=False),
                              fig_height=FloatSlider(min=3., max=15., step=.2, value=fig_height, continuous_update=False),
                              use_TeX=use_TeX, 
                              font_size=IntSlider(min=6, max=16, step=1, value=font_size, continuous_update=False))
    
    # Construct save figure button
    save_fig_button = interactive(save_current_subplots, {"manual":True, "manual_name":"save figure"},
                          fig_directory=fig_directory, 
                          fig_name=fig_name)
    
    return plot_fig_button, save_fig_button


def save_plots_interface(my_plot, 
                    fig_size=(6.4, 4.8), use_TeX=False, font_size=10,
                    fig_directory="./figs/", fig_name="FIG", plot_on_call=False):
    # NOTE: default plotting parameters are matplotlib default options

    # Plot figure on the call of this function with passed parameters
    if plot_on_call:
        fig_width, fig_height = fig_size
        plot_figure(my_plot, fig_width, fig_height, use_TeX, font_size)

    # Get plot figure and save figure button and display them
    plot_fig_button, save_fig_button = get_save_plots_interface(my_plot, 
                                                                fig_size, use_TeX, font_size, 
                                                                fig_directory, fig_name)
    display(plot_fig_button, save_fig_button)




############################
# Save variables interface #
############################
# TODO: use pickle or np.savez instead of np.save
def save_variables_as_dict(variable_names="var1, var2", output_directory="./npys/", output_name="OUTPUT", save_separately=False):

    # Assert output_name does not include further directories path or other improper chars
    forbidden_chars = ["/", "\\", ":", "\"", "?"] # backslash is "\\" or chr(92)
    assert all([forbidden_char not in output_name for forbidden_char in forbidden_chars]), f"output_name should not contain these chars: {forbidden_chars}"
    
    # Set output directory 
    Path(output_directory).mkdir(parents=True, exist_ok=True)

    # Split along "," and strip potential extra-spaces 
    variable_names_list = [unstriped.strip() for unstriped in variable_names.split(",")]

    print("Saving...", end="\r")
    if save_separately:
        # Save each variable separately in a dict 
        # e.g. {name1:value1} in file OUTPUT_name1, {name2:value2} in file OUTPUT_name2
        for variable_name in variable_names_list:
            dict_to_save = dict()
            variable_value = __dict__[variable_name] # Access global variable
            dict_to_save[variable_name] = variable_value
            output_fullname = join(output_directory, output_name + "_" + variable_name)
            np.save(output_fullname, dict_to_save)
            print(f"Saved {output_fullname}.npy")

        print(f"Saved variables in {output_directory} folder. To load last variable for instance, type:")
        loading_hint_str = f"""```\n{output_name} = np.load("{output_fullname}.npy", allow_pickle=True).item()\n"""
        loading_hint_str += variable_name + f", = {output_name}.values()\n```" 
        # NOTE: in `variable, = dict.values()` the comma is crucial to handle single entry dict
        print(loading_hint_str)

    else:
        # Save all variables in a dict
        # e.g. {name1:value1, name2_value2} in file OUTPUT
        dict_to_save = dict()
        for variable_name in variable_names_list:
            variable_value = __dict__[variable_name] # Access global variable
            dict_to_save[variable_name] = variable_value

        output_fullname = join(output_directory, output_name)
        np.save(output_fullname, dict_to_save)
        print(f"Saved variables in {output_directory} folder. To load, type:")
        loading_hint_str = f"""```\n{output_name} = np.load("{output_fullname}.npy", allow_pickle=True).item()\n"""

        for variable_name in variable_names_list:
            loading_hint_str += variable_name + ", "
        loading_hint_str = loading_hint_str + f"= {output_name}.values()\n```"
        # In `variable, = dict.values()` the comma is crucial to handle single entry dict
        print(loading_hint_str)


def get_save_variables_interface(variable_names="var1, var2", output_directory="./npys/", output_name="OUTPUT", save_separately=False):
    
    # Construct save variables button
    save_variables_button = interactive(save_variables_as_dict, {"manual":True, "manual_name":"save variables"},
                                        variable_names=variable_names,
                                        output_directory=output_directory, 
                                        output_name=output_name,
                                        save_separately=save_separately)
    return save_variables_button

def save_variables_interface(variable_names="var1, var2", output_directory="./npys/", output_name="OUTPUT", save_separately=False, save_on_call=False):
    
    # Save variables on the call of this function with passed parameters
    if save_on_call:
        save_variables_as_dict(variable_names, output_directory, output_name, save_separately)
    
    # Get save variables button and display it
    save_variables_button = get_save_variables_interface(variable_names, output_directory, output_name, save_separately)
    display(save_variables_button)













