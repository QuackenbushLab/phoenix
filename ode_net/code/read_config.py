# Move this into train
import configparser


def read_arguments_from_file(fp):
    """Reads run arguments from file"""
    config = configparser.ConfigParser()
    config.read(fp)

    settings = config['settings']

    return _convert_arguments(settings)

def _convert_arguments(settings):
    converted_settings = {}
    converted_settings['viz'] = settings.getboolean('viz')
    converted_settings['viz_every_iteration'] = False
    converted_settings['verbose'] = True
    converted_settings['method'] = settings['method']
    converted_settings['neurons_per_layer'] = settings.getint('neurons_per_layer')
    converted_settings['optimizer'] = settings['optimizer']

    converted_settings['batch_type'] = settings['batch_type']
    converted_settings['batch_size'] = settings.getint('batch_size')
    converted_settings['batch_time'] = 99999
    converted_settings['batch_time_frac'] = 99999

    converted_settings['init_lr'] = settings.getfloat('init_lr')
    converted_settings['weight_decay'] = settings.getfloat('weight_decay')
    converted_settings['dec_lr'] = False
    converted_settings['dec_lr_factor'] = 999

    converted_settings['cpu'] = settings.getboolean('cpu')
    converted_settings['val_split'] = settings.getfloat('val_split')
    converted_settings['noise'] = settings.getfloat('noise')
    converted_settings['epochs'] = settings.getint('epochs')

    converted_settings['solve_eq_gridsize'] = 100
    converted_settings['solve_A'] = False

    converted_settings['debug'] = False  
    converted_settings['output_dir'] = "output"
    converted_settings['normalize_data'] = settings.getboolean('normalize_data')  
    converted_settings['explicit_time'] = False
    converted_settings['relative_error'] = False

    converted_settings['pretrained_model'] = settings.getboolean('pretrained_model')   
    converted_settings['lr_range_test'] = False 
    converted_settings['scale_expression'] = settings.getfloat('scale_expression')
    converted_settings['log_scale'] = 'linear'
    converted_settings['init_bias_y'] = 0

    
    
    
    
    return converted_settings
