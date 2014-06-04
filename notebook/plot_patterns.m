% PLOT_PATTERNS(filepath):

function plot_patterns(filepath)
    pattern_struct = load(filepath);
    names = fieldnames(pattern_struct);
    patterns = struct2cell(pattern_struct);
    nb_comp = size(patterns);
    nb_comp = nb_comp(1);
    for i=1:nb_comp
        name = names(i);
        name = name{1};
        plot_brain_signal(name, cell2mat(patterns(i)));
    end

function plot_brain_signal(name, sig_values, varargin)
    lims = [min(sig_values), max(sig_values)];
    if nargin == 3
        lims = varargin{1};
    end

    add_view = 1;
    sig_len = length(sig_values);

    data_path = '/Users/kikohs/work/MATLAB/brain/code';
    out_folder = '/Users/kikohs/work/pro/dybrain/notebook/data/brain3d/';

    % Set paths for surface and label files
    path_surf_rh = strcat(data_path, '/plot_kit/needed_data/bert_FREESURFER/surf/rh.pial');
    path_surf_lh = strcat(data_path, '/plot_kit/needed_data/bert_FREESURFER/surf/lh.pial');
    if sig_len == 1000
        base_path_annot_rh = strcat(data_path, '/plot_kit/needed_data/bert_FREESURFER/label/rh.myaparc');
        base_path_annot_lh = strcat(data_path, '/plot_kit/needed_data/bert_FREESURFER/label/lh.myaparc');
    elseif sig_len == 448
        path_annot_rh = strcat(data_path, '/plot_kit/needed_data/bert_FREESURFER/label/rh.myaparc_250.annot');
        path_annot_lh = strcat(data_path, '/plot_kit/needed_data/bert_FREESURFER/label/lh.myaparc_250.annot');
        scale = 4;
    elseif sig_len == 219
        path_annot_rh = strcat(data_path, '/plot_kit/needed_data/bert_FREESURFER/label/rh.myaparc_125.annot');
        path_annot_lh = strcat(data_path, '/plot_kit/needed_data/bert_FREESURFER/label/lh.myaparc_125.annot');
        scale = 3;
    elseif sig_len == 114
        path_annot_rh = strcat(data_path, '/plot_kit/needed_data/bert_FREESURFER/label/rh.myaparc_60.annot');
        path_annot_lh = strcat(data_path, '/plot_kit/needed_data/bert_FREESURFER/label/lh.myaparc_60.annot');
        scale = 2;
    elseif sig_len == 68
        path_annot_rh = strcat(data_path, '/plot_kit/needed_data/bert_FREESURFER/label/rh.myaparc_36.annot');
        path_annot_lh = strcat(data_path, '/plot_kit/needed_data/bert_FREESURFER/label/lh.myaparc_36.annot');
        scale = 1;
    else
        error('Supported signal sizes (# areas on brain): 68, 114, 219, 448, 1000');
    end

    % Load necessary data
    load(strcat(data_path, '/plot_kit/needed_data/labels_index_CORTICAL_Laus2008_all_scales.mat'));
    % Choose a colormap (e.g. 'jet')
    map = colormap(cbrewer('seq', 'YlOrRd', 9));
    CM = mapsc2rgb_2(sig_values, map, lims);% right hemisphere nodes will be mapped to the colormap

    % Get figure handles
    [h, j, k] = colorsurf_2hemi(path_surf_rh, path_annot_rh, path_surf_lh, path_annot_lh, add_view, CM, llist{scale});
    save_fig(name, 1, [90, 0], h, out_folder)
    save_fig(name, 2, [-90, 0], h, out_folder)
    save_fig(name, 3, [90, 0], j, out_folder)
    save_fig(name, 4, [-90, 0], j, out_folder)
    save_fig(name, 5, [0, 90], k, out_folder)
    close all


function save_fig(name, subid, viewpoint, handle, out_folder)
    figure(handle);
    view(viewpoint);
    camlight; % creates a light right and up from camera
    lighting gouraud; % specify lighting algorithm
    path = strcat(out_folder, name, '_', num2str(subid));
%     print(handle, '-depsc', path)
%     print(handle, '-dpng', path)
    export_fig(path, '-png', '-transparent', handle);