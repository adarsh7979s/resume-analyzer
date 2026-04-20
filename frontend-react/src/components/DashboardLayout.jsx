import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  LayoutDashboard, Search, BarChart3, Clock, Settings,
  LogOut, Menu, X, Zap,
} from 'lucide-react';
import './DashboardLayout.css';

const NAV_ITEMS = [
  { id: 'dashboard', label: 'Dashboard',  icon: <LayoutDashboard size={18} /> },
  { id: 'analyzer',  label: 'Analyzer',   icon: <Search size={18} /> },
  { id: 'insights',  label: 'Insights',   icon: <BarChart3 size={18} /> },
  { id: 'history',   label: 'History',     icon: <Clock size={18} /> },
];

const FOOTER_NAV = [
  { id: 'settings', label: 'Settings', icon: <Settings size={18} /> },
  { id: 'logout',   label: 'Logout',   icon: <LogOut size={18} /> },
];

export default function DashboardLayout({ children, activeTab, setActiveTab, candidateName }) {
  const [mobileOpen, setMobileOpen] = useState(false);

  const initials = candidateName
    ? candidateName.split(' ').map(w => w[0]).join('').toUpperCase().slice(0, 2)
    : 'RA';

  const headerTitle = NAV_ITEMS.find(n => n.id === activeTab)?.label || 'Dashboard';

  return (
    <div className="dash-shell">
      {/* ── Mobile toggle ── */}
      <button
        className="dash-mobile-toggle"
        onClick={() => setMobileOpen(o => !o)}
        aria-label="Toggle sidebar"
      >
        {mobileOpen ? <X size={20} /> : <Menu size={20} />}
      </button>

      {/* ── Mobile overlay ── */}
      <div
        className={`dash-sidebar-overlay ${mobileOpen ? 'visible' : ''}`}
        onClick={() => setMobileOpen(false)}
      />

      {/* ── Sidebar ── */}
      <aside className={`dash-sidebar ${mobileOpen ? 'open' : ''}`}>
        <div className="dash-sidebar-brand">
          <div className="dash-brand-icon">
            <Zap size={18} />
          </div>
          ResuMatch AI
        </div>

        <nav className="dash-sidebar-nav">
          {NAV_ITEMS.map(item => (
            <button
              key={item.id}
              className={`dash-nav-item ${activeTab === item.id ? 'active' : ''}`}
              onClick={() => { setActiveTab(item.id); setMobileOpen(false); }}
            >
              {item.icon}
              {item.label}
            </button>
          ))}
        </nav>

        <div className="dash-sidebar-footer">
          {FOOTER_NAV.map(item => (
            <button
              key={item.id}
              className={`dash-nav-item ${activeTab === item.id ? 'active' : ''}`}
              onClick={() => { setActiveTab(item.id); setMobileOpen(false); }}
            >
              {item.icon}
              {item.label}
            </button>
          ))}
        </div>
      </aside>

      {/* ── Header ── */}
      <header className="dash-header">
        <h1 className="dash-header-title">{headerTitle}</h1>

        <div className="dash-header-right">
          <button className="dash-header-btn" aria-label="Search">
            <Search size={16} />
          </button>

          <div className="dash-header-profile">
            <div className="dash-header-avatar">{initials}</div>
            <span className="dash-header-profile-name">
              {candidateName || 'Guest'}
            </span>
          </div>
        </div>
      </header>

      {/* ── Main content ── */}
      <main className="dash-main">
        <AnimatePresence mode="wait">
          <motion.div
            key={activeTab}
            initial={{ opacity: 0, y: 12 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -8 }}
            transition={{ duration: 0.35, ease: [0.16, 1, 0.3, 1] }}
          >
            {children}
          </motion.div>
        </AnimatePresence>
      </main>
    </div>
  );
}
