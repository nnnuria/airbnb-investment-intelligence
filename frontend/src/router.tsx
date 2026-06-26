import { createBrowserRouter } from "react-router-dom";
import { AppShell } from "@/components/layout/AppShell";
import Dashboard from "@/pages/Dashboard";
import NewAnalysis from "@/pages/NewAnalysis";
import Workspace from "@/pages/Workspace";
import Saved from "@/pages/Saved";

export const router = createBrowserRouter([
  {
    element: <AppShell />,
    children: [
      { path: "/", element: <Dashboard /> },
      { path: "/new", element: <NewAnalysis /> },
      { path: "/workspace", element: <Workspace /> },
      { path: "/saved", element: <Saved /> },
    ],
  },
]);
