from typing import TYPE_CHECKING

from . import Config


if TYPE_CHECKING:
    from src import wrapper, models, load_data


class VertexConfig(Config['models.AutoEncoderVertex','wrapper.VertexWrapper', 'load_data.AutoEncoderVertexV2']):
    construction_axis: int = 3
    sample_count_per_vertex: int = 2000
    positional_encoding: bool = True
