# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

__all__ = ["build_head"]


def build_head(config, **kwargs):
    # det head
    from .det_db_head import DBHead, PFHeadLocal # noqa
    from .det_east_head import EASTHead # noqa
    from .det_sast_head import SASTHead # noqa
    from .det_pse_head import PSEHead # noqa
    from .det_fce_head import FCEHead # noqa

    # rec head
    from .rec_ctc_head import CTCHead # noqa
    from .rec_att_head import AttentionHead # noqa
    from .rec_srn_head import SRNHead # noqa
    from .rec_nrtr_head import Transformer # noqa
    from .rec_sar_head import SARHead # noqa
    from .rec_can_head import CANHead # noqa
    from .rec_multi_head import MultiHead # noqa

    # cls head
    from .cls_head import ClsHead # noqa

    support_dict = [
        "DBHead",
        "PSEHead",
        "EASTHead",
        "SASTHead",
        "CTCHead",
        "ClsHead",
        "AttentionHead",
        "SRNHead",
        "PGHead",
        "Transformer",
        "TableAttentionHead",
        "SARHead",
        "FCEHead",
        "CANHead",
        "MultiHead",
        "PFHeadLocal",
    ]

    module_name = config.pop("name")
    assert module_name in support_dict, Exception(
        "head only support {}".format(support_dict)
    )
    module_class = eval(module_name)(**config, **kwargs)
    return module_class
