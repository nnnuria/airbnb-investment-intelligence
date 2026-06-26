import { Outlet } from "react-router-dom";
import { AppBar } from "./AppBar";
import { Footer } from "./Footer";
import { CopilotPanel } from "@/features/copilot/CopilotPanel";

export function AppShell() {
  return (
    <div className="flex min-h-screen flex-col bg-background">
      <AppBar />
      <main className="mx-auto w-full max-w-page flex-1 px-7 py-8">
        <Outlet />
      </main>
      <Footer />
      <CopilotPanel />
    </div>
  );
}
