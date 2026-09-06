import { panel } from "../dashboard/panel";
import { sample, robots } from "./samples";
import type { RobotState } from "./types";

export function workbench() {
  const shared = panel();
  const selected = document.getElementById("robot-select") as HTMLSelectElement;
  const armPanel = document.getElementById("arm-panel")!;
  let last: RobotState | null = null;
  let names: string[] = [];
  shared.onSelect(() => {
    armPanel.hidden = selected.value !== "arm";
  });
  function state(value: RobotState, record: boolean) {
    last = value;
    shared.state(sample(value, names), record);
    if (value.arm && !document.getElementById("joints")!.children.length) {
      value.arm.actuator_names.forEach((name, i) => {
        const label = document.createElement("label");
        label.textContent = name.replace("arm/", "");
        const input = document.createElement("input");
        input.type = "number";
        input.setAttribute("aria-label", name);
        input.min = String(value.arm!.control_limits[i][0]);
        input.max = String(value.arm!.control_limits[i][1]);
        input.step = "any";
        input.value = String(value.arm!.control[i]);
        label.append(input);
        const reading = document.createElement("output");
        reading.id = `joint-reading-${i}`;
        label.append(reading);
        document.getElementById("joints")!.append(label);
      });
    }
    value.arm?.actuator_names.forEach((_, i) => {
      if (!record)
        document.querySelectorAll<HTMLInputElement>("#joints input")[i].value =
          String(value.arm!.control[i]);
      document.getElementById(`joint-reading-${i}`)!.textContent =
        i < 7
          ? `q ${value.arm!.position_rad[i].toFixed(3)} rad · ${value.arm!.effort_nm[i].toFixed(2)} Nm`
          : `actuator effort ${value.arm!.actuator_force[i].toFixed(2)} (source units)`;
    });
  }
  return {
    state,
    setup(ids: string[]) {
      names = ids;
      last = null;
      shared.setup(
        robots(ids),
        ids.map((id) => ({
          id,
          label: `${id} · raw RGB`,
          width: 512,
          height: 384,
        })),
      );
      document.getElementById("joints")!.replaceChildren();
    },
    armValues() {
      return Array.from(
        document.querySelectorAll<HTMLInputElement>("#joints input"),
        (input) => {
          if (!input.value || !input.checkValidity())
            throw Error(
              `Enter a valid setpoint for ${input.getAttribute("aria-label")}`,
            );
          return Number(input.value);
        },
      );
    },
    captureLabel(text: string) {
      document.getElementById("capture-time")!.textContent = text;
    },
    selection() {
      return selected.value;
    },
    last() {
      return last;
    },
  };
}
