from multiprocessing.pool import ThreadPool
import numpy as np
import vigra
from ..io.infer_loader import SimpleParallelLoader
from ..utils.various import yaml2dict
try:
    import inferno
    from inferno.io.volumetric import VolumeLoader
except ImportError:
    inferno = None


def process_batch(batches, dataset, get_slicings, segmentation_pipeline):
    assert len(batches) == 1
    assert len(batches[0]) == 1
    index = batches[0][0]
    input_ = dataset[index]
    print("[+] Processing block {} of {}.".format(index + 1, len(dataset)))
    # print("[*] Input-shape {}".format(input_.shape))

    # get the slicings w.r.t. the current prediction and the output
    global_slicing_incl_pad = dataset.base_sequence[index]
    local_slicing, global_slicing = get_slicings(global_slicing_incl_pad,
                                                      input_.shape,
                                                      dataset.padding)
    # remove offset dim from slicing
    global_slicing_incl_pad = global_slicing_incl_pad[1:]
    global_slicing = global_slicing[1:]
    local_slicing = local_slicing[1:]

    # print("Global slice: {}".format(global_slicing))
    outputs = segmentation_pipeline(input_)

    if isinstance(outputs, tuple):
        assert len(outputs) == 2
        output_patch = outputs[0]
        extra_outputs = outputs[1]
        extra_outputs_cropped = extra_outputs[local_slicing]
    else:
        output_patch = outputs
        extra_outputs = None
        extra_outputs_cropped = None


    output_patch, max_label_patch, _ = vigra.analysis.relabelConsecutive(output_patch.astype(np.uint32),
                                                                         keep_zeros=False)

    # Run connected components after cropping the padding:
    output_patch_cropped = vigra.analysis.labelVolume(output_patch[local_slicing].astype(np.uint32))

    return [output_patch, output_patch_cropped, extra_outputs, extra_outputs_cropped,
            global_slicing_incl_pad, global_slicing, max_label_patch]


class BlockWise(object):
    def __init__(self, segmentation_pipeline,
                 offsets,
                 blockwise=True,
                 final_agglomerater=None,
                 invert_affinities=False, # Only used for final aggl.
                 nb_threads=8,
                 return_fragments=False,
                 blockwise_config=None):
        if inferno is None:
            raise ImportError("Inferno is required to run BlockWise")
        self.segmentation_pipeline = segmentation_pipeline
        self.blockwise = blockwise
        self.return_fragments = return_fragments
        if blockwise:
            assert blockwise_config is not None
            self.blockwise_solver = BlockWiseSegmentationPipelineSolver.from_config(
                segmentation_pipeline,
                offsets,
                blockwise_config)

            # At the moment the final agglomeration is a usual  HC with 0.5 threshold:
            # if final_agglomerater is None:
            #     raise DeprecationWarning()
            #     self.final_agglomerater = GreedyEdgeContractionAgglomeraterFromSuperpixels(
            #         offsets,
            #         max_distance_lifted_edges=2,
            #         update_rule_merge='mean',
            #         update_rule_not_merge='mean',
            #         zero_init=False,
            #         n_threads=nb_threads,
            #         invert_affinities=invert_affinities)
            # else:
            self.final_agglomerater = final_agglomerater

    def __call__(self, *inputs_):
        assert len(inputs_) == 1 or len(inputs_) == 2
        input_ = inputs_[0]
        # final_crop = tuple(slice(pad[0], input_.volume.shape[i+1] - pad[1]) for i, pad in enumerate(input_.padding[1:]))
        if self.blockwise:
            assert len(inputs_) == 1, "Input segmentation not supported with blockwise"
            assert isinstance(input_, VolumeLoader)
            # TODO: change this!!
            # At the moment if we crop the padding, then we need to crop the global border for the final
            # agglomeration (but in this way we lose affinities context).
            # The alternative is to keep somehow the segmentation on the borders...
            blockwise_segm = self.blockwise_solver(input_)
            if self.return_fragments:
                raise DeprecationWarning()
                fragments = blockwise_segm[0]
                blockwise_segm = blockwise_segm[1]

            # # ---- TEMP ----
            # blockwise_segm = blockwise_segm[final_crop]
            # affs = input_.volume[(slice(None),) + final_crop]
            # # ---- TEMP ----

            if self.final_agglomerater is not None:
                raise DeprecationWarning("The second output is not supported. Please update")
                output_segm = self.final_agglomerater(input_.volume, blockwise_segm)
            else:
                output_segm = blockwise_segm
            # output_segm = output_segm[final_crop]
            # blockwise_segm = blockwise_segm[final_crop]
            if self.return_fragments:
                raise DeprecationWarning("Make it general for a general second output")
                return output_segm, blockwise_segm, fragments
            else:
                return output_segm
        else:
            if isinstance(input_, VolumeLoader):
                input_ = input_.volume
            # if isinstance(inputs_[1], VolumeLoader):
            #     inputs_[1] = inputs_[1].volume
            if len(inputs_) == 1:
                output_segm = self.segmentation_pipeline(input_)
            elif len(inputs_) == 2:
                output_segm = self.segmentation_pipeline(input_, inputs_[1])
            else:
                raise NotImplementedError()
            if self.return_fragments:
                raise DeprecationWarning("Make it general for a general second output")
                return output_segm[1], output_segm[0]
            else:
                return output_segm


class BlockWiseSegmentationPipelineSolver(object):
    def __init__(self,
                 segmentation_pipeline,
                 crop_padding=False,
                 offsets=None,
                 nb_parallel_blocks=1,
                 nb_threads=8,
                 num_workers=1):
        """
        :param blockwise: if False, the whole dataset is processed together
        :param nb_parallel_blocks: how many blocks are solved in parallel
        :param nb_threads: nb threads used computations in every block
        :param num_workers: used to load affinities from file (probably not needed, since there is no augmentation)
        """
        self.segmentation_pipeline = segmentation_pipeline
        self.nb_offsets = len(offsets)

        if offsets is not None:
            assert len(offsets) == self.nb_offsets, "%i, %i" % (len(offsets), self.nb_offsets)
        self.offsets = offsets

        self.crop_padding = crop_padding

        self.nb_parallel_blocks = nb_parallel_blocks
        # TODO: not necessary!
        self.nb_threads = nb_threads
        self.num_workers = num_workers


    @classmethod
    def from_config(cls, segmentation_pipeline, offsets, config):
        config = yaml2dict(config)
        crop_padding = config.get("crop_padding", False)
        nb_threads = config.get("nb_threads", 8)
        nb_parallel_blocks = config.get("nb_parallel_blocks", 1)
        num_workers = config.get("num_workers", 1)
        if offsets is not None:
            offsets = [tuple(off) for off in offsets]
        return cls(segmentation_pipeline,
                   crop_padding=crop_padding,
                   nb_threads=nb_threads,
                   nb_parallel_blocks=nb_parallel_blocks,
                   num_workers=num_workers,
                   offsets=offsets)


    def __call__(self, dataset):
        # build the output volume
        shape_affs = dataset.volume.shape
        assert shape_affs[0] == self.nb_offsets
        assert len(shape_affs) == 4
        shape_output = shape_affs[1:]

        output = np.zeros(shape_output, dtype='uint64')
        output_padded = np.zeros(shape_output, dtype='uint64')
        extra_output = None
        extra_output_padded = None

        # loader
        loader = SimpleParallelLoader(dataset, num_workers=self.num_workers, enqueue_samples=False)
        # mask to count the number of times a pixel was inferred





        max_label = 0
        if self.nb_parallel_blocks == 1:
            while True:
                batches = loader.next_batch()
                if not batches:
                    print("[*] All blocks were processed!")
                    break

                outputs = process_batch(batches, dataset, self.get_slicings, self.segmentation_pipeline)
                output_patch, output_patch_cropped, extra_outputs_patch, extra_outputs_patch_cropped, global_slicing_incl_pad, global_slicing, max_label_patch = tuple(outputs)

                # Save padded output:
                output_padded[global_slicing_incl_pad] = output_patch + max_label
                output[global_slicing] = output_patch_cropped + max_label

                # EXTRA OUTPUT:
                # The extra volume should have shape (volume_shape, nb_channels)
                if extra_outputs_patch is not None:
                    if extra_output is None:
                        extra_output = np.zeros(shape_output + (extra_outputs_patch.shape[-1], ), dtype='uint64')
                        extra_output_padded = np.zeros(shape_output + (extra_outputs_patch.shape[-1], ), dtype='uint64')
                    extra_output[global_slicing] = extra_outputs_patch_cropped
                    extra_output_padded[global_slicing_incl_pad] = extra_outputs_patch

                max_label += max_label_patch + 1
        else:
            pool = ThreadPool(processes=self.nb_parallel_blocks)
            from itertools import repeat
            output_patch, output_patch_cropped, extra_outputs_patch, extra_outputs_patch_cropped, global_slicing_incl_pad, global_slicing, max_label_patch = zip(*pool.starmap(process_batch,
                                    zip(loader,
                                        repeat(dataset),
                                        repeat(self.get_slicings),
                                        repeat(self.segmentation_pipeline))))
            pool.close()
            pool.join()

            # Save and combine predictions:
            for i in range(len(output_patch)):
                output_padded[global_slicing_incl_pad[i]] = output_patch[i] + max_label
                output[global_slicing[i]] = output_patch_cropped[i] + max_label
                max_label += max_label_patch[i] + 1

                # EXTRA OUTPUT:
                # The extra volume should have shape (volume_shape, nb_channels)
                if extra_outputs_patch is not None:
                    assert extra_outputs_patch[i].shape[:-1] == output_patch[i].shape

                    if extra_output is None:
                        extra_output = np.zeros(shape_output + (extra_outputs_patch[i].shape[-1], ), dtype='uint64')
                        extra_output_padded = np.zeros(shape_output + (extra_outputs_patch[i].shape[-1], ), dtype='uint64')
                    extra_output[global_slicing[i]] = extra_outputs_patch_cropped[i]
                    extra_output_padded[global_slicing_incl_pad[i]] = extra_outputs_patch[i]


        print("Out shape", output.shape)
        print("Out padded shape", output_padded.shape)
        # Combine padded output with cropped one:
        final_output = output
        final_extra_output = extra_output
        if any(tuple([pad[0]!=0  for i, pad in enumerate(dataset.padding)])):
            print("Combining padded outputs:")
            global_pad = tuple(slice(pad[0], dataset.volume.shape[i] - pad[1])
                                   for i, pad in enumerate(dataset.padding))
            final_output = output_padded + max_label
            # print("Final out shape", final_output.shape)
            # print("Out shape", output.shape)
            # print("Out shape cropped", output[global_pad[1:]].shape)
            final_output[global_pad[1:]] = output[global_pad[1:]]
            final_output, _, _ = vigra.analysis.relabelConsecutive(final_output,
                                                                                 keep_zeros=False)

            if extra_output is not None:
                final_extra_output = extra_output_padded
                final_extra_output[global_pad[1:]] = extra_output[global_pad[1:]]

            if not dataset.volume_already_padded:
                final_output = final_output[global_pad[1:]]
                if extra_output is not None:
                    final_extra_output = final_extra_output[global_pad[1:]]

        print("Done!")

        # # crop padding from the outputs
        # crop = tuple(slice(pad[0], shape_output[i] - pad[1]) for i, pad in enumerate(dataset.padding[1:]))
        # FIXME: why this...?
        # output[output==-1] = max_label + 1
        # return the prediction (not cropped)

        # print("Final blocks: max --> {}, min --> {}".format(output.max(), output.min()))
        # output = vigra.analysis.labelVolume(output.astype(np.uint32))
        # print("Final blocks: max --> {}, min --> {}".format(output.max(), output.min()))
        return final_output, final_extra_output


    def get_slicings(self, slicing, shape, padding):
        # crop away the padding (we treat global as local padding) if specified
        # this is generally not necessary if we use blending
        if self.crop_padding:
            # slicing w.r.t the current output
            local_slicing = tuple(slice(pad[0], shape[i] - pad[1])
                                  for i, pad in enumerate(padding))
            # slicing w.r.t the global output
            global_slicing = tuple(slice(slicing[i].start + pad[0],
                                         slicing[i].stop - pad[1])
                                   for i, pad in enumerate(padding))
        # otherwise do not crop
        else:
            local_slicing = np.s_[:]
            global_slicing = slicing
        return local_slicing, global_slicing



