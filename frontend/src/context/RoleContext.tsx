import { createContext, useContext, useState } from 'react'

export type Role = 'admin' | 'therapist' | 'user' | null

interface RoleContextValue {
  role: Role
  setRole: (r: Role) => void
  clearRole: () => void
}

const RoleContext = createContext<RoleContextValue>({
  role: null,
  setRole: () => {},
  clearRole: () => {},
})

export function RoleProvider({ children }: { children: React.ReactNode }) {
  const [role, setRoleState] = useState<Role>(() => {
    return (localStorage.getItem('psych_role') as Role) ?? null
  })

  function setRole(r: Role) {
    setRoleState(r)
    if (r) localStorage.setItem('psych_role', r)
    else localStorage.removeItem('psych_role')
  }

  function clearRole() {
    setRoleState(null)
    localStorage.removeItem('psych_role')
  }

  return (
    <RoleContext.Provider value={{ role, setRole, clearRole }}>
      {children}
    </RoleContext.Provider>
  )
}

export const useRole = () => useContext(RoleContext)
