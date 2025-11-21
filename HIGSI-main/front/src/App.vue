<script setup>
import Header from './components/Header.vue';
import { onMounted, ref } from 'vue';
import { useLoginStore } from "@/stores/LoginStore";
import { useThemeStore } from "@/stores/ThemeStore";
import user_img from "@/assets/static/imgs/avatar.svg"
import { useRouter } from 'vue-router';

const router = useRouter()

function logout() {
  login.logoutAction();
  router.push({ name: 'login' })
}
//import { useLoginStore, useThemeStore } from '@/stores';
// Variável reativa para o tema
const login = useLoginStore()
const theme = useThemeStore()

onMounted(() => {
  login.loginVerific()
  theme.verify()
})

</script>

<template>
 
  <div class=" h-100">
    <div class="row container-login g-0 h-100">
      <div class="col-2 header sticky-top "
        v-if="$route.path !== '/login' && $route.path !== '/sing-in' && $route.path !== '/seleciona_unidade' && $route.name !== 'not-found'">
        <Header>
        </Header>

      </div>
      <div class="col"
        :class="$route.path !== '/login' && $route.path !== '/sing-in' && $route.path !== '/seleciona_unidade' && $route.name !== 'not-found' ? `px-4 pt-4` : ``">
        <div v-if="$route.path !== '/login' && $route.path !== '/sing-in' && $route.path !== '/seleciona_unidade' && $route.name !== 'not-found'"
          class="d-flex justify-content-between ">
          <h1 v-if="$route.name === 'home'">Histórico</h1>
          <h1 v-if="$route.name === 'dashboard'">Dashboard</h1>
          <div class="d-flex align-items-center gap-2">
            <button @click="logout" class="d-flex align-items-center gap-1 button-logout">
              <img class="w-4" :src="user_img" alt="">
              <span class="fs-2">log-out</span>
            </button>
            <!-- Botão para mudar para o tema escuro -->
            <button class="nav-link px-0 hide-theme-dark" title="Habilitar modo escuro" data-bs-toggle="tooltip"
              data-bs-placement="bottom" @click="theme.setTheme('dark')">
              <span v-html="svgs.darkThemeIcon"></span>
            </button>

            <!-- Botão para mudar para o tema claro -->
            <button class="nav-link px-0 hide-theme-light" title="Habilitar modo claro" data-bs-toggle="tooltip"
              data-bs-placement="bottom" @click="theme.setTheme('light')">
              <span v-html="svgs.lightThemeIcon"></span>
            </button>
          </div>

        </div>
        <RouterView />
      </div>
     
    </div>
  </div>


</template>

<style scoped>
.button-logout {
  background: none;
  border: none;
}
.header{

}
.modal-overlay {
  position: fixed;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background: rgba(0, 0, 0, 0.5);
  display: flex;
  align-items: center;
  justify-content: center;
}

.container-login {
  min-height: 100vh;
  max-width: 100%;
  overflow-y: scroll;
}

.modal-dialog {
  background: white;
  padding: 20px;
  border-radius: 8px;
  width: 400px;
  max-width: 90%;
}
</style>
