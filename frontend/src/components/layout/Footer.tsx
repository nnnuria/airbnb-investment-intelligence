export function Footer() {
  return (
    <footer className="mt-16 border-t border-hairline">
      <div className="mx-auto flex max-w-page flex-col gap-2 px-7 py-6 text-[12px] text-foggy sm:flex-row sm:items-center sm:justify-between">
        <p className="max-w-2xl leading-relaxed">
          Estimates only — not investment, tax, or legal advice. Figures are
          modelled from market data and the assumptions you provide, which you
          can edit at any time.
        </p>
        <p className="shrink-0">
          Modelling powered by{" "}
          <span className="font-extrabold text-kpmg-deep">KPMG</span>
        </p>
      </div>
    </footer>
  );
}
