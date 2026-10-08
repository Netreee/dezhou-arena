import type { ActionOption } from "./models";

export function toCliLine(option: ActionOption, total?: number): string {
  if (option.kind !== "bet_to" && option.kind !== "raise_to") return option.kind;
  if (total === undefined || !Number.isInteger(total)) throw new Error("请输入整数金额");
  if (option.min_to === null || option.max_to === null || total < option.min_to || total > option.max_to) {
    throw new Error("金额应在服务器提供的范围内");
  }
  return `${option.kind} ${total}`;
}
