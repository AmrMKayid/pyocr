from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import copy

__all__ = ["build_post_process"]


def build_post_process(config, global_config=None):
    from .db_postprocess import DBPostProcess # noqa
    from .east_postprocess import EASTPostProcess # noqa
    from .fce_postprocess import FCEPostProcess # noqa
    from .rec_postprocess import ( # noqa
        CTCLabelDecode,
        AttnLabelDecode,
        SRNLabelDecode,
        TableLabelDecode,
        NRTRLabelDecode,
        SARLabelDecode,
        ViTSTRLabelDecode,
        RFLLabelDecode,
    )
    from .cls_postprocess import ClsPostProcess # noqa
    from .rec_postprocess import CANLabelDecode # noqa

    support_dict = [
        "DBPostProcess",
        "EASTPostProcess",
        "SASTPostProcess",
        "CTCLabelDecode",
        "AttnLabelDecode",
        "ClsPostProcess",
        "SRNLabelDecode",
        "PGPostProcess",
        "TableLabelDecode",
        "NRTRLabelDecode",
        "SARLabelDecode",
        "FCEPostProcess",
        "ViTSTRLabelDecode",
        "CANLabelDecode",
        "RFLLabelDecode",
    ]

    if config["name"] == "PSEPostProcess":
        from .pse_postprocess import PSEPostProcess # noqa

        support_dict.append("PSEPostProcess")

    config = copy.deepcopy(config)
    module_name = config.pop("name")
    if global_config is not None:
        config.update(global_config)
    assert module_name in support_dict, Exception(
        "post process only support {}, but got {}".format(support_dict, module_name)
    )
    module_class = eval(module_name)(**config)
    return module_class
