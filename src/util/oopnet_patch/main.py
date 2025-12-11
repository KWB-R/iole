from collections import defaultdict

import oopnet
import logging

from oopnet.utils.getters import (
    get_pattern,
    get_junction,
)
from oopnet.reader.decorators import section_reader

logger = logging.getLogger(__name__)

@section_reader("DEMANDS", 2)
def read_demands(network, block: list):
    """
    Fixed parser; old version will overwrite the "main" demand and
    map the demand multiplier to the wrong pattern if base demand
    for a category is 0
    """

    logger.debug("Reading demand section")
    print(f"Demand read with modified demand reader.")

    _pattern_ids = set()
    _multipliers = defaultdict(list)
    _patterns = defaultdict(list)

    for b in block:
        vals = b["values"]

        junction_id = vals[0]
        multiplier = float(vals[1])
        pattern_id = vals[2]

        # update pattern set
        _pattern_ids.add(pattern_id)

        # collect all junction-related info
        _multipliers[junction_id].append(multiplier)
        _patterns[junction_id].append(pattern_id)

    # retrieve patterns
    patterns = {p: get_pattern(network, id=p) for p in _pattern_ids}

    # set junction attrs
    for jid, mult in _multipliers.items():
        pat = _patterns.get(jid, None)

        j = get_junction(network, id=jid)

        # set demands with list of multipliers (mult will always be a list now)
        j.demand = mult

        if pat:
            j.demandpattern = [patterns[pid] for pid in pat]

# monkey patch
oopnet.reader.reading_modules.read_system_operation.read_demands = read_demands
print("Monkey patch to oopnet.reader.reading_modules.read_system_operation.read_demands applied.")
