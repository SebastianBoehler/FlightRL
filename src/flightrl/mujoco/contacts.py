from __future__ import annotations


def forbidden_contact_count(model, data, *, vehicle_body_id: int) -> int:
    """Count contacts between the vehicle subtree and the static scene."""
    count = 0
    vehicle_root_id = int(model.body_rootid[vehicle_body_id])
    for index in range(int(data.ncon)):
        contact = data.contact[index]
        first_is_vehicle = _geom_root_id(model, int(contact.geom1)) == vehicle_root_id
        second_is_vehicle = _geom_root_id(model, int(contact.geom2)) == vehicle_root_id
        count += int(first_is_vehicle != second_is_vehicle)
    return count


def _geom_root_id(model, geom_id: int) -> int:
    body_id = int(model.geom_bodyid[geom_id])
    return int(model.body_rootid[body_id])
