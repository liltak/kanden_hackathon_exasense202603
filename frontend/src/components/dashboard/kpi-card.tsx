interface KPICardProps {
  title: string;
  value: string;
  unit: string;
  gradient: string;
}

const GRADIENTS: Record<string, string> = {
  blue: "from-sky-500 to-blue-400",
  green: "from-emerald-500 to-green-400",
  orange: "from-amber-500 to-yellow-400",
  red: "from-rose-500 to-orange-400",
  purple: "from-violet-500 to-purple-400",
  default: "from-gray-500 to-gray-400",
};

export function KPICard({ title, value, unit, gradient }: KPICardProps) {
  const grad = GRADIENTS[gradient] || GRADIENTS.default;
  return (
    <div className={`rounded-xl bg-gradient-to-br ${grad} p-5 text-white`}>
      <h3 className="text-xs font-medium opacity-90">{title}</h3>
      <div className="mt-2 text-3xl font-bold">{value}</div>
      <div className="mt-1 text-xs opacity-80">{unit}</div>
    </div>
  );
}
