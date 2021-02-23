# This file provides factories for different postprocessing pipelines loaded according to a config file

from .WS import WatershedOnDistanceTransformFromAffinities, MutexWatershed, WatershedFromAffinities
from .multicut import Multicut, MulticutPipelineFromAffinities
from GASP.affinities import AccumulatorLongRangeAffs
from GASP.segmentation.GASP.run_from_affinities import GaspFromAffinities, SegmentationFeeder

from copy import deepcopy

def get_superpixel_generator(postproc_kwargs, offsets, invert_affinities, nb_threads):
    superpixel_gen = None
    if postproc_kwargs.get('from_superpixels', False):
        superpixel_generator_type = postproc_kwargs.get('superpixel_generator_type', None)
        if superpixel_generator_type == 'WSDT':
            WSDT_kwargs = deepcopy(postproc_kwargs.get('WSDT_kwargs', {}))
            superpixel_gen = WatershedOnDistanceTransformFromAffinities(
                offsets,
                WSDT_kwargs.pop('threshold', 0.5),
                WSDT_kwargs.pop('sigma_seeds', 0.),
                invert_affinities=invert_affinities,
                return_hmap=False,
                n_threads=nb_threads,
                **WSDT_kwargs,
                **postproc_kwargs.get('prob_map_kwargs', {}))
        elif superpixel_generator_type == 'WS':
            superpixel_gen = WatershedFromAffinities(
                offsets,
                return_hmap=False,
                invert_affinities=invert_affinities,
                n_threads=nb_threads,
                **postproc_kwargs.get('prob_map_kwargs', {}))
        elif postproc_kwargs.get("start_from_given_segmentation"):
            superpixel_gen = SegmentationFeeder()
        else:
            raise NotImplementedError("The current superpixel_generator_type was not recognised: {}".format(superpixel_generator_type))
    return superpixel_gen


def get_segmentation_pipeline(
        segm_pipeline_type,
        offsets,
        nb_threads=1,
        invert_affinities=False,
        **post_proc_config):

    multicut_kwargs = post_proc_config.get('multicut_kwargs', {})
    MWS_kwargs = post_proc_config.get('MWS_kwargs', {})
    GASP_kwargs = post_proc_config.get('GASP_kwargs', {})

    if segm_pipeline_type == 'only_superpixel_generator':
        post_proc_config['from_superpixels'] = True
    elif post_proc_config.get('start_from_given_segmentation', False):
        post_proc_config['from_superpixels'] = False

    superpixel_generator = get_superpixel_generator(post_proc_config,
                                          offsets,
                                          invert_affinities,
                                          nb_threads)

    if segm_pipeline_type == 'only_superpixel_generator':
        segm_pipeline = superpixel_generator
    elif segm_pipeline_type == 'multicut':
        featurer = AccumulatorLongRangeAffs(offsets, n_threads=nb_threads,
                                         offsets_weights=multicut_kwargs.get('offsets_weights'),
                                         used_offsets=multicut_kwargs.get('used_offsets'),
                                         invert_affinities= not invert_affinities,
                                         verbose=False,
                                         return_dict=True,
                                         offset_probabilities=multicut_kwargs.get('offsets_probabilities', 1.0)
                                         )
        if post_proc_config.get('start_from_given_segm', False):
            segm_pipeline = Multicut(featurer, edge_statistic='mean',
                                     beta=multicut_kwargs.get('beta', 0.5),
                                     weighting_scheme=multicut_kwargs.get('weighting_scheme', None),
                                     weight=multicut_kwargs.get('weight', 16.),
                                     time_limit=multicut_kwargs.get('time_limit', None),
                                     solver_type=multicut_kwargs.get('solver_type', 'kernighanLin'),
                                     verbose_visitNth=multicut_kwargs.get('verbose_visitNth', 100000000),
                                     )
        else:
            assert post_proc_config['from_superpixels'], "A superpixel_generator is needed for multicut"
            segm_pipeline = MulticutPipelineFromAffinities(superpixel_generator, featurer, edge_statistic='mean',
                                                           beta=multicut_kwargs.get('beta', 0.5),
                                                           weighting_scheme=multicut_kwargs.get('weighting_scheme',
                                                                                                None),
                                                           weight=multicut_kwargs.get('weight', 16.),
                                                           time_limit=multicut_kwargs.get('time_limit', None),
                                                           solver_type=multicut_kwargs.get('solver_type',
                                                                                           'kernighanLin'),
                                                           verbose_visitNth=multicut_kwargs.get('verbose_visitNth',
                                                                                                100000000))
    elif segm_pipeline_type == 'GASP':
        # ------------------------------
        # Build agglomeration:
        # ------------------------------
        segm_pipeline = GaspFromAffinities(
            offsets,
            superpixel_generator=superpixel_generator,
            n_threads=nb_threads,
            invert_affinities=invert_affinities,
            **GASP_kwargs
        )

    elif segm_pipeline_type == 'MWS':
        segm_pipeline = MutexWatershed(offsets,
                                     invert_affinities=invert_affinities,
                                   **MWS_kwargs)
    else:
        raise NotImplementedError("The passed segmentation pipeline type was not recognised: {}".format(segm_pipeline_type))
    return segm_pipeline

