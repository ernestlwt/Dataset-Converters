from past.builtins import basestring

formats = [
    'TDG',
    'TDGSEGM',
    'SSD',
    'FRCNN',
    'ADE20K',
    'ADE20K2021',
    'CITYSCAPES',
    'COCO',
    'VOC',
    'VOCCALIB',
    'VOCSEGM',
    'OID',
    'CVAT',
    'YOLO',
    'YOLO2021'
]


def gen_conversion_format(input_format, output_format):

    if not isinstance(input_format, basestring):
        raise Exception('Input format is not a string')

    if not isinstance(output_format, basestring):
        raise Exception('Output format is not a string')

    if not input_format in formats:
        raise Exception('Unsupported input format: ' + input_format)

    if not output_format in formats:
        raise Exception('Unsupported output format: ' + output_format)

    return input_format + '2' + output_format
