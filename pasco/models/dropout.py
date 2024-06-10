import torch
import MinkowskiEngine as ME


class MinkowskiIdentity(ME.MinkowskiModuleBase):

    def __init__(self, *args, **kwargs):
        super(MinkowskiIdentity, self).__init__()

    def forward(self, input):
        return input


class MinkowskiNonlinearityBase(ME.MinkowskiModuleBase):
    MODULE = None

    def __init__(self, *args, **kwargs):
        super(MinkowskiNonlinearityBase, self).__init__()
        self.module = self.MODULE(*args, **kwargs)

    def forward(self, input):
        output = self.module(input.F)
        if isinstance(input, ME.TensorField):
            return ME.TensorField(
                output,
                coordinate_field_map_key=input.coordinate_field_map_key,
                coordinate_manager=input.coordinate_manager,
                quantization_mode=input.quantization_mode,
            )
        else:
            return ME.SparseTensor(
                output,
                coordinate_map_key=input.coordinate_map_key,
                coordinate_manager=input.coordinate_manager,
            )


class MinkowskiLocationDropout(MinkowskiNonlinearityBase):
    MODULE = torch.nn.Dropout1d


class MinkowskiSpatialDropout(MinkowskiNonlinearityBase):
    MODULE = torch.nn.Dropout1d

    def forward(self, input):
        output = self.module(input.F.T).T
        if isinstance(input, ME.TensorField):
            return ME.TensorField(
                output,
                coordinate_field_map_key=input.coordinate_field_map_key,
                coordinate_manager=input.coordinate_manager,
                quantization_mode=input.quantization_mode,
            )
        else:
            return ME.SparseTensor(
                output,
                coordinate_map_key=input.coordinate_map_key,
                coordinate_manager=input.coordinate_manager,
            )


if __name__ == "__main__":
    coords = torch.randint(9, (10, 4))
    feats = torch.randn(10, 5)

    stensor1 = ME.SparseTensor(feats, coordinates=coords)
    stensor2 = MinkowskiSpatialDropout(p=0.5)(stensor1)
    import pdb

    pdb.set_trace()
    print()
