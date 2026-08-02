"""Adapter tests: native record -> Message, with correct ground-truth labels.

Fixtures are tiny inline samples in each dataset's native format, so the tests
run with no downloaded data.
"""
from vis.shared.contracts import Bus
from vis.shared.datasets import (
    OTIDS_DOS_PREDICATE,
    CanMirguSource,
    CarHackingSource,
    OtidsSource,
    RoadSource,
    VeReMiSource,
)


def test_car_hacking_variable_width_and_flags(tmp_path):
    # Headerless, variable width: DLC drives the number of data columns; the
    # last column is the R/T flag.
    csv_text = (
        "1478198376.389427,0316,8,05,21,68,09,21,21,00,6f,R\n"
        "1478198376.389511,0000,8,00,00,00,00,00,00,00,00,T\n"
        "1478198376.389600,02a0,2,12,34,R\n"   # short DLC, flag still last
    )
    p = tmp_path / "DoS_dataset.csv"
    p.write_text(csv_text)

    msgs = list(CarHackingSource(p, attack_type="dos"))
    assert len(msgs) == 3
    assert msgs[0].arbitration_id == 0x316
    assert msgs[0].data == bytes.fromhex("052168092121006f")
    assert msgs[0].is_attack is False
    assert msgs[1].is_attack is True and msgs[1].attack_type == "dos"
    # short-DLC row parsed correctly (2 data bytes, benign)
    assert msgs[2].data == bytes([0x12, 0x34]) and msgs[2].is_attack is False


def test_otids_candump_text_and_predicate(tmp_path):
    text = (
        "Timestamp: 1479121434.850202        ID: 0260    000    DLC: 8    19 21 22 30 08 8e 6d 3a\n"
        "Timestamp: 1479121434.850300        ID: 0000    000    DLC: 8    00 00 00 00 00 00 00 00\n"
    )
    p = tmp_path / "DoS_attack_dataset.txt"
    p.write_text(text)

    # No attack_type => attack-free interpretation, everything benign.
    benign = list(OtidsSource(p))
    assert [m.is_attack for m in benign] == [False, False]
    assert benign[0].arbitration_id == 0x260
    assert benign[0].data == bytes.fromhex("19212230088e6d3a")

    # DoS scenario: the 0x000 flood frame is the injected one.
    dos = list(OtidsSource(p, attack_type="dos", inject_predicate=OTIDS_DOS_PREDICATE))
    assert [m.is_attack for m in dos] == [False, True]
    assert dos[1].attack_type == "dos"


def test_road_masquerade_labeled_by_metadata_window(tmp_path):
    # candump .log; injection id 0x0D0 is a *legitimate* id (masquerade), so
    # only the frame inside the injection_interval is an attack.
    log = (
        "(100.0) can0 0D0#FFFF0000000000A8\n"   # rel t=0.0  -> benign
        "(105.0) can0 0D0#0000000000000000\n"   # rel t=5.0  -> in window, attack
        "(106.0) can0 1F0#DEADBEEF\n"           # other id, in window -> benign
        "(120.0) can0 0D0#0102030405060708\n"   # rel t=20.0 -> after window, benign
    )
    p = tmp_path / "masquerade.log"
    p.write_text(log)

    meta = {"injection_id": "0x0D0", "injection_interval": [4.0, 10.0]}
    msgs = list(RoadSource(p, attack_type="max_speedometer_masquerade", metadata=meta))
    assert [m.is_attack for m in msgs] == [False, True, False, False]
    assert msgs[1].attack_type == "max_speedometer_masquerade"
    assert msgs[1].arbitration_id == 0x0D0


def test_can_mirgu_header_autodetect_and_labels(tmp_path):
    # Canonical header, mixed data formats, Label 0/1 with a Category name.
    csv_text = (
        "Timestamp,Arbitration_ID,DLC,Data,Label,Category\n"
        "1690000000.1,0x18f,8,00 11 22 33 44 55 66 77,0,normal\n"
        "1690000000.2,0x2c0,8,ffffffffffffffff,1,masquerade\n"
    )
    p = tmp_path / "attack.csv"
    p.write_text(csv_text)

    msgs = list(CanMirguSource(p))
    assert [m.is_attack for m in msgs] == [False, True]
    assert msgs[0].arbitration_id == 0x18F
    assert msgs[0].data == bytes(range(0x00, 0x78, 0x11))  # 00 11 22 .. 77
    assert msgs[1].data == b"\xff" * 8
    assert msgs[1].attack_type == "masquerade"   # taken from Category column


def test_can_mirgu_alias_header_and_dlc_trim(tmp_path):
    # Different spellings + a short DLC that should trim the data.
    csv_text = (
        "time,CAN ID,dlc,payload,attack\n"
        "1.0,0x100,2,de ad be ef,1\n"   # dlc=2 -> only de ad kept
    )
    p = tmp_path / "benign_alias.csv"
    p.write_text(csv_text)

    (msg,) = list(CanMirguSource(p, attack_type="fabrication"))
    assert msg.arbitration_id == 0x100
    assert msg.data == bytes([0xDE, 0xAD])
    assert msg.is_attack and msg.attack_type == "fabrication"  # no Category -> fallback


def test_veremi_embedded_and_ground_truth_labels(tmp_path):
    # Receiver log: a type-2 own-GPS (skipped), a benign BSM, an embedded
    # attacker BSM, and one whose label comes from the GroundTruth file.
    log = "\n".join([
        '{"type":2,"rcvTime":100.0,"pos":[0,0,0]}',
        '{"type":3,"rcvTime":100.1,"sender":11,"messageID":1,"pos":[10,20,0],"spd":[5,0,0],"attackerType":0}',
        '{"type":3,"rcvTime":100.2,"sender":12,"messageID":2,"pos":[99,99,0],"spd":[0,0,0],"attackerType":2}',
        '{"type":3,"rcvTime":100.3,"sender":13,"messageID":3,"pos":[1,2,0],"spd":[1,1,0]}',
    ])
    p = tmp_path / "JSONlog-7.json"
    p.write_text(log)
    gt = tmp_path / "GroundTruthJSONlog.json"
    gt.write_text('{"messageID":3,"attackerType":16}')

    msgs = list(VeReMiSource(p, ground_truth=gt))
    assert len(msgs) == 3                       # type-2 record skipped
    assert all(m.bus == Bus.V2X for m in msgs)
    assert [m.is_attack for m in msgs] == [False, True, True]
    assert msgs[1].attack_type == "const_pos_offset"   # embedded attackerType 2
    assert msgs[2].attack_type == "eventual_stop"      # from ground-truth file (16)
    assert msgs[0].arbitration_id == 11
    # kinematics are carried in the structured payload field
    assert msgs[0].payload["pos"] == (10.0, 20.0, 0.0)
    assert msgs[0].payload["spd"] == (5.0, 0.0, 0.0)
    assert msgs[0].data == b""                         # V2X frames use payload, not data


def test_can_mirgu_candump_with_trailing_label(tmp_path):
    # the REAL CAN-MIRGU format: candump + per-frame label (0 benign / 1 attack),
    # variable-length payloads, no DLC column.
    log = (
        "(1683206351.761563) can0 421#000000FFE37F0065 0\n"
        "(1683206351.761564) can0 2B0#EE03000751 0\n"          # 5-byte payload
        "(1698319591.695248) can0 000#FFFFFFFFFFFFFFFF 1\n"    # injected
    )
    p = tmp_path / "DoS_attack.log"
    p.write_text(log)

    msgs = list(CanMirguSource(p, attack_type="dos"))
    assert [m.is_attack for m in msgs] == [False, False, True]
    assert msgs[0].arbitration_id == 0x421
    assert msgs[1].data == bytes.fromhex("EE03000751")          # variable length preserved
    assert msgs[2].attack_type == "dos"


def test_road_fuzzing_placeholder_id_degrades_instead_of_crashing(tmp_path):
    # ROAD fuzzing captures use "XXX" because injected ids are random; the
    # adapter must not crash and must not invent labels.
    p = tmp_path / "fuzzing_attack_1.log"
    p.write_text("(100.0) can0 0D0#FFFF0000000000A8\n(101.0) can0 1F0#DEADBEEF\n")
    msgs = list(RoadSource(p, metadata={"injection_id": "XXX",
                                        "injection_interval": [0.0, 10.0]}))
    assert len(msgs) == 2
    assert all(not m.is_attack for m in msgs)     # no id-based ground truth
