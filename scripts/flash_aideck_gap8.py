from __future__ import annotations

import argparse
import hashlib
import time
from pathlib import Path

import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.mem import MemoryElement
from cflib.crazyflie.mem.deck_memory import SyncDeckMemoryManager
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from cflib.utils.power_switch import PowerSwitch


CONFIRM_TOKEN = "FLASH_AIDECK_GAP8"
DECK_NAME = "bcAI:gap8"


def find_deck(decks, name: str):
    matches = [(index, deck) for index, deck in decks.items() if deck.name == name]
    if len(matches) != 1:
        found = ", ".join(deck.name for deck in decks.values()) or "none"
        raise RuntimeError(f"expected one {name} deck, found: {found}")
    return matches[0]


def wait_for_state(manager, deck_index: int, *, bootloader: bool, timeout_s: float):
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        deck = manager.query_decks().get(deck_index)
        if deck is not None and deck.is_started and deck.is_bootloader_active == bootloader:
            return deck
        time.sleep(0.25)
    state = "bootloader" if bootloader else "firmware"
    raise TimeoutError(f"{DECK_NAME} did not enter {state} mode")


def flash(image: Path, uri: str, reboot_uri: str, cache_dir: Path) -> None:
    payload = image.read_bytes()
    if not payload:
        raise ValueError(f"image is empty: {image}")
    digest = hashlib.sha256(payload).hexdigest()
    cache_dir.mkdir(parents=True, exist_ok=True)

    cflib.crtp.init_drivers()
    crazyflie = Crazyflie(rw_cache=str(cache_dir))
    with SyncCrazyflie(uri, cf=crazyflie) as scf:
        memories = scf.cf.mem.get_mems(MemoryElement.TYPE_DECK_MEMORY)
        if len(memories) != 1:
            raise RuntimeError(f"expected one deck-memory manager, found {len(memories)}")
        manager = SyncDeckMemoryManager(memories[0])
        deck_index, deck = find_deck(manager.query_decks(), DECK_NAME)

        if not deck.supports_fw_upgrade:
            raise RuntimeError(f"{DECK_NAME} does not support firmware upgrades")
        if not deck.supports_reset_to_bootloader:
            raise RuntimeError(f"{DECK_NAME} does not support bootloader reset")

        print(f"image={image} bytes={len(payload)} sha256={digest}")
        deck.reset_to_bootloader()
        deck = wait_for_state(manager, deck_index, bootloader=True, timeout_s=5.0)
        deck.set_fw_new_flash_size(len(payload))

        last_percent = -1

        def progress(_message: str, percent: int) -> None:
            nonlocal last_percent
            if percent != last_percent:
                print(f"write={percent}%", flush=True)
                last_percent = percent

        if not deck.write_sync(0, payload, progress):
            raise RuntimeError(f"failed to write {DECK_NAME}")

        if deck.supports_reset_to_fw:
            deck.reset_to_fw()
            wait_for_state(manager, deck_index, bootloader=False, timeout_s=8.0)

    if not deck.supports_reset_to_fw:
        switch = PowerSwitch(reboot_uri)
        try:
            switch.stm_power_cycle()
        finally:
            switch.close()
        time.sleep(4)

    print(f"flashed={DECK_NAME} uri={uri} reboot_uri={reboot_uri} sha256={digest}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Flash the AI Deck GAP8 through the normal firmware link")
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--uri", required=True)
    parser.add_argument("--reboot-uri", required=True)
    parser.add_argument("--cache-dir", type=Path, default=Path("artifacts/cflib_cache"))
    parser.add_argument("--confirm", required=True)
    args = parser.parse_args()

    if args.confirm != CONFIRM_TOKEN:
        parser.error(f"--confirm must be {CONFIRM_TOKEN}")
    flash(args.image, args.uri, args.reboot_uri, args.cache_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
