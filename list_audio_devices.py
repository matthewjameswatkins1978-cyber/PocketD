"""List available audio input devices for the diagnostic live-onset test."""

import sounddevice as sd


def main() -> None:
    print("Available audio devices:")
    for index, device in enumerate(sd.query_devices()):
        channels = int(device.get("max_input_channels", 0))
        print(f"[{index}] {device.get('name')} | inputs={channels}")


if __name__ == "__main__":
    main()
