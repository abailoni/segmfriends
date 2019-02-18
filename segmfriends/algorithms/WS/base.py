class WatershedBase(object):
    def __init__(self, grower):
        # check that this is callable
        self.grower = grower

    def __call__(self, affinities):
        return self.grower(affinities)

    @staticmethod
    def get_2d_from_3d_offsets(offsets):
        # only keep in-plane channels
        keep_channels = [ii for ii, off in enumerate(offsets) if off[0] == 0]
        offsets = [off[1:] for ii, off in enumerate(offsets) if ii in keep_channels]
        return keep_channels, offsets
