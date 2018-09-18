from scene3d import log


def main():
    log.debug('global logger debug')
    log.info('global logger info')
    log.warning('global logger warning')

    logger = log.make_logger('my_logger', level=log.DEBUG)
    log.add_stream_handler(logger, level=log.INFO)
    log.add_file_handler(logger, '/tmp/scene3d_test/test_logging.log', level=log.DEBUG)
    logger.debug('my logger debug')
    logger.info('my logger info')
    logger.warning('my logger warning')


if __name__ == '__main__':
    main()
