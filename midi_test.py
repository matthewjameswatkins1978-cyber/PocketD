import time
import mido

PORT_NAME = "PocketDrummer Out"

KICK = 36
SNARE = 38
HAT = 42

BPM = 120
BEAT_SECONDS = 60 / BPM


def find_port() -> str:
    ports = mido.get_output_names()
    print("Available MIDI outputs:")
    for port in ports:
        print(" -", port)

    for port in ports:
        if PORT_NAME.lower() in port.lower():
            return port

    raise RuntimeError(f"Could not find MIDI port containing: {PORT_NAME}")


def hit(port: mido.ports.BaseOutput, note: int, velocity: int = 100, length: float = 0.05) -> None:
    port.send(mido.Message("note_on", note=note, velocity=velocity))
    time.sleep(length)
    port.send(mido.Message("note_off", note=note, velocity=0))


def main() -> None:
    output_name = find_port()
    print(f"\nUsing MIDI output: {output_name}")
    print("Playing test groove. Press Ctrl+C to stop.\n")

    with mido.open_output(output_name) as port:
        while True:
            # Beat 1
            hit(port, KICK, 110)
            hit(port, HAT, 70)
            time.sleep(BEAT_SECONDS - 0.05)

            # Beat 2
            hit(port, SNARE, 110)
            hit(port, HAT, 70)
            time.sleep(BEAT_SECONDS - 0.05)

            # Beat 3
            hit(port, KICK, 105)
            hit(port, HAT, 70)
            time.sleep(BEAT_SECONDS - 0.05)

            # Beat 4
            hit(port, SNARE, 110)
            hit(port, HAT, 70)
            time.sleep(BEAT_SECONDS - 0.05)


if __name__ == "__main__":
    main()
