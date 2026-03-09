import { BrowserRouter, Routes, Route, Outlet, Navigate, useLocation } from 'react-router-dom'
import { AuthProvider, useAuth } from './context/AuthContext'
import Layout from './components/Layout'
import ErrorBoundary from './components/ErrorBoundary'
import LandingPage from './pages/LandingPage'
import Dashboard from './pages/Dashboard'
import LiveAnalysis from './pages/LiveAnalysis'
import Users from './pages/Users'
import Analytics from './pages/Analytics'
import SessionHistory from './pages/SessionHistory'
import Alerts from './pages/Alerts'
import Admin      from './pages/Admin'
import Therapist  from './pages/Therapist'
import UserPortal from './pages/UserPortal'

/** Wraps portal routes: requires authentication + enforces role-based access */
function PortalLayout() {
  const { user } = useAuth()
  const { pathname } = useLocation()

  if (!user) return <Navigate to="/" replace />

  // Redirect to correct portal if accessing wrong one
  if (pathname === '/admin' && user.role !== 'admin')
    return <Navigate to={user.role === 'therapist' ? '/therapist' : '/user'} replace />
  if (pathname === '/therapist' && user.role !== 'therapist')
    return <Navigate to={user.role === 'admin' ? '/admin' : '/user'} replace />
  if (pathname === '/user' && user.role !== 'user')
    return <Navigate to={user.role === 'admin' ? '/admin' : '/therapist'} replace />

  return (
    <Layout>
      <ErrorBoundary>
        <Outlet />
      </ErrorBoundary>
    </Layout>
  )
}

export default function App() {
  return (
    <BrowserRouter>
      <AuthProvider>
        <Routes>
          {/* Landing / Login — no sidebar */}
          <Route path="/" element={<LandingPage />} />

          {/* Authenticated portal pages — wrapped in sidebar Layout */}
          <Route element={<PortalLayout />}>
            <Route path="/dashboard"  element={<Dashboard />} />
            <Route path="/live"       element={<LiveAnalysis />} />
            <Route path="/users"      element={<Users />} />
            <Route path="/analytics"  element={<Analytics />} />
            <Route path="/history"    element={<SessionHistory />} />
            <Route path="/alerts"     element={<Alerts />} />
            <Route path="/admin"      element={<Admin />} />
            <Route path="/therapist"  element={<Therapist />} />
            <Route path="/user"       element={<UserPortal />} />
          </Route>
        </Routes>
      </AuthProvider>
    </BrowserRouter>
  )
}
