# this file provides factories for different postprocessing pipelines

from .WS import WatershedOnDistanceTransformFromAffinities, MutexWatershed, WatershedFromAffinities
from .agglo import GreedyEdgeContractionAgglomeraterFromSuperpixels, GreedyEdgeContractionClustering
from .multicut import Multicut, MulticutPipelineFromAffinities
from ..features.featurer import FeaturerLongRangeAffs

from copy import deepcopy

def get_fragmented(postproc_kwargs, offsets, invert_affinities, nb_threads):
    fragmenter = None
    if postproc_kwargs.get('use_fragmenter', False):
        assert 'fragmenter' in postproc_kwargs
        fragm_type = postproc_kwargs['fragmenter']
        if fragm_type == 'WSDT':
            WSDT_kwargs = deepcopy(postproc_kwargs.get('WSDT_kwargs', {}))
            fragmenter = WatershedOnDistanceTransformFromAffinities(
                offsets,
                WSDT_kwargs.pop('threshold', 0.5),
                WSDT_kwargs.pop('sigma_seeds', 0.),
                invert_affinities=invert_affinities,
                return_hmap=False,
                n_threads=nb_threads,
                **WSDT_kwargs,
                **postproc_kwargs.get('prob_map_kwargs', {}))
        elif fragm_type == 'WS':
            fragmenter = WatershedFromAffinities(
                offsets,
                return_hmap=False,
                invert_affinities=invert_affinities,
                n_threads=nb_threads,
                **postproc_kwargs.get('prob_map_kwargs', {}))
        else:
            raise NotImplementedError()
    return fragmenter


def get_segmentation_pipeline(
        segm_pipeline_type,
        offsets,
        nb_threads=1,
        invert_affinities=False,
        return_fragments=False,
        **post_proc_config):

    multicut_kwargs = post_proc_config.get('multicut_kwargs', {})
    MWS_kwargs = post_proc_config.get('MWS_kwargs', {})
    generalized_HC_kwargs = post_proc_config.get('generalized_HC_kwargs', {})

    if segm_pipeline_type == 'only_fragmenter':
        post_proc_config['use_fragmenter'] = True
    elif post_proc_config.get('start_from_given_segm', False):
        post_proc_config['use_fragmenter'] = False

    fragmenter = get_fragmented(post_proc_config,
                                    offsets,
                                    invert_affinities,
                                    nb_threads)

    if segm_pipeline_type == 'only_fragmenter':
        segm_pipeline = fragmenter
    elif segm_pipeline_type == 'multicut':

        featurer = FeaturerLongRangeAffs(offsets, n_threads=nb_threads,
                                         offsets_weights=multicut_kwargs.get('offsets_weights'),
                                         used_offsets=multicut_kwargs.get('used_offsets'),
                                         invert_affinities= not invert_affinities,
                                         debug=False,
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
            assert post_proc_config['use_fragmenter'], "A fragmenter is needed for multicut"
            segm_pipeline = MulticutPipelineFromAffinities(fragmenter, featurer, edge_statistic='mean',
                                                           beta=multicut_kwargs.get('beta', 0.5),
                                                           weighting_scheme=multicut_kwargs.get('weighting_scheme',
                                                                                                None),
                                                           weight=multicut_kwargs.get('weight', 16.),
                                                           time_limit=multicut_kwargs.get('time_limit', None),
                                                           solver_type=multicut_kwargs.get('solver_type',
                                                                                           'kernighanLin'),
                                                           verbose_visitNth=multicut_kwargs.get('verbose_visitNth',
                                                                                                100000000))


    elif segm_pipeline_type == 'gen_HC':
        HC_kwargs = generalized_HC_kwargs

        if not post_proc_config.get('start_from_given_segm', False):
            # ------------------------------
            # Build agglomeration:
            # ------------------------------
            segm_pipeline = GreedyEdgeContractionClustering(
                offsets,
                fragmenter,
                n_threads=nb_threads,
                invert_affinities=invert_affinities,
                return_fragments=return_fragments,
                **HC_kwargs.get('agglomeration_kwargs', {})
            )
        else:
            segm_pipeline = GreedyEdgeContractionAgglomeraterFromSuperpixels(
                offsets,
                n_threads=nb_threads,
                invert_affinities=invert_affinities,
                **HC_kwargs.get('agglomeration_kwargs', {})
            )


    elif segm_pipeline_type == 'MWS':
        segm_pipeline = MutexWatershed(offsets,
                                     # min_segment_size=10,
                                     invert_affinities=invert_affinities,
                                   n_threads=nb_threads,
                                   **MWS_kwargs)
    else:
        raise NotImplementedError()
    return segm_pipeline

