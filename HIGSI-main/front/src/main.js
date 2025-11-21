import './assets/main.css'
import { AxiosPlugin } from 'v-api-fetch';
// tabler
import '@tabler/core/dist/css/tabler.min.css'
import '@tabler/core/dist/js/tabler.min.js'
// toast
import Toast from "vue-toastification";
import "vue-toastification/dist/index.css";
import { useLoginStore } from "@/stores/LoginStore";
import vRequired from "v-required"
const options = {
    // You can set your default options here
};
import { createApp } from 'vue'
import { createPinia } from 'pinia'

import VueApexCharts from "vue3-apexcharts";

import App from './App.vue'

import router from './router'

const app = createApp(App)



import * as svgs from '@/assets/static/js/svgs.js'; // Importando todos os SVGs

// Adicionando os SVGs ao protótipo global
app.config.globalProperties.svgs = svgs;

const pinia = createPinia();
app.use(pinia)
app.use(router)
app.directive("required", vRequired);

app.use(Toast, options);
app.use(VueApexCharts);
app.use(AxiosPlugin, {
  hostApi: 'http://localhost:8000',
  // hostApi: 'https://backend-oncoia.bentricode.com',
  baseApi: '/api',

  requestInterceptor: (config) => {
    const loginStore = useLoginStore();
    if (loginStore.token) {
      // Adiciona o cabeçalho 'Authorization' em todas as chamadas
      config.headers["Authorization"] = `Bearer ${loginStore.token}`;
    }
    return config;
  },

});
pinia.use(({ store }) => {
  store.$app = app; 
  store.$api = app.config.globalProperties.$api; // Injeta a instância $api
});
app.mount('#app')
