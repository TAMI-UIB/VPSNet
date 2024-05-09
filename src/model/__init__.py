from .vpsnet import VPSNet
from .vpsnet_post import VPSNetPost
from .vpsnet_learned import VPSNetLearned
from .vpsnet_learned_pgcu import VPSNetLearnedPGCU
from .vpsnet_memory_concat import VPSNetConcatMemory
from .vpsnet_concatenated_proxnet import VPSNetConcat
from .vpsnet_learned_memory import VPSNetLearnedMemory
from .vpsnet_learned_malisat import VPSNetLearnedMalisat
from .vpsnet_learned_memory_malisat import VPSNetLearnedMemoryMalisat
from .vpsnet_learned_malisat_radiometric import VPSNetLearnedMalisatRadiometric


dict_model = {
    "VPSNet": VPSNet,
    "VPSNetPost": VPSNetPost,
    "VPSNetLearned": VPSNetLearned,
    "VPSNetMalisat": VPSNetLearnedMalisat,
    "VPSNetMalisatRadiometric": VPSNetLearnedMalisatRadiometric,
    "VPSNetPGCU": VPSNetLearnedPGCU,
    "VPSNetMemory": VPSNetLearnedMemory,
    "VPSNetConcat": VPSNetConcat,
    "VPSNetConcatMemory": VPSNetConcatMemory,
    "VPSNetLearnedMemoryMalisat": VPSNetLearnedMemoryMalisat,
}

