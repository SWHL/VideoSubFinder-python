g_im_save_format = ".jpeg"

g_mthr = 0.4
g_smthr = 0.25
g_mnthr = 0.3
g_segw = 8
g_segh = 3
g_msegc = 2
g_scd = 800
g_btd = 0.05
g_to = 0.1  # Max Text Offset

g_mpn = 50
g_mpd = 0.3
g_msh = 0.01
g_msd = 0.2
g_mpned = 0.3

g_min_alpha_color = 1

g_dmaxy = 8

# / min-max posize for resolution ~ 480p=640×480 scaled to x4 == 4-18p (1-4.5p in original size)
g_minpw = 4.0 / 640.0
g_maxpw = 18.0 / 640.0
g_minph = 4.0 / 480.0
g_maxph = 18.0 / 480.0
g_minpwh = 2.0 / 3.0

g_min_dI = 9
g_min_dQ = 9
g_min_ddI = 14
g_min_ddQ = 14

g_scale = 4

# define STR_SIZE (256 * 2)

# define MAX_EDGE_STR (11 * 16 * 256)

g_use_ILA_images_for_getting_txt_symbols_areas = False
g_use_ILA_images_before_clear_txt_images_from_borders = False
g_use_ILA_images_for_clear_txt_images = True
g_use_gradient_images_for_clear_txt_images = True
g_clear_txt_images_by_main_color = True

g_clear_image_logical = False

g_generate_cleared_text_images_on_test = False
g_show_results = False
g_show_sf_results = False
g_clear_test_images_folder = True
g_show_transformed_images_only = False

g_wxImageHandlersInitialized = False

g_use_ocl = True

g_use_cuda_gpu = True

g_dL_color = 40
g_dA_color = 30
g_dB_color = 30

g_combine_to_single_cluster = False

g_cuda_kmeans_initial_loop_iterations = 10
g_cuda_kmeans_loop_iterations = 30
g_cpu_kmeans_initial_loop_iterations = 10
g_cpu_kmeans_loop_iterations = 10

g_min_h = 12.0 / 720.0  # ~ min sub height in percents to image height

g_remove_wide_symbols = True

g_disable_save_images = False

g_save_each_substring_separately = True
g_save_scaled_images = True

g_border_is_darker = True

g_extend_by_grey_color = False
g_allow_min_luminance = 100
