import { create } from 'zustand';
import Keycloak from 'keycloak-js';
import { webSocketManager } from '../services/api';

interface AuthState {
  keycloak: Keycloak | null;
  isAuthenticated: boolean;
  user: any;
  token: string | null;
  initialize: () => Promise<void>;
  login: () => void;
  logout: () => void;
}

export const useAuthStore = create<AuthState>((set, get) => ({
  keycloak: null,
  isAuthenticated: false,
  user: null,
  token: null,

  initialize: async () => {
    const keycloak = new Keycloak({
      url: 'http://keycloak.local:8080/auth',
      realm: '254carbon',
      clientId: 'web-hub',
    });

    try {
      const authenticated = await keycloak.init({
        onLoad: 'check-sso',
        checkLoginIframe: false,
      });

      set({
        keycloak,
        isAuthenticated: authenticated,
        token: keycloak.token || null,
        user: authenticated ? keycloak.tokenParsed : null,
      });

      // Refresh token periodically
      if (authenticated) {
        setInterval(() => {
          keycloak.updateToken(70).catch(() => {
            console.error('Failed to refresh token');
          });
          // Notify websocket manager to re-send subscription with refreshed token
          webSocketManager.resubscribe();
        }, 60000);
      }
    } catch (error) {
      console.error('Failed to initialize Keycloak', error);
    }
  },

  login: () => {
    const { keycloak } = get();
    keycloak?.login();
  },

  logout: () => {
    const { keycloak } = get();
    keycloak?.logout();
    set({ isAuthenticated: false, user: null, token: null });
  },
}));

