<script setup>
import logo from '@/assets/static/imgs/onco-logo (1).png';
import { ref, onMounted } from 'vue';

import { useLoginStore } from "@/stores/LoginStore";
import { useThemeStore } from "@/stores/ThemeStore";
import { useToastStore } from "@/stores/useToastStore";
import { useRouter } from 'vue-router';
import fundo_login from "@/assets/static/imgs/fundologin.png"
//import { useLoginStore, useThemeStore } from '@/stores';
// Variável reativa para o tema
const login = useLoginStore()
const theme = useThemeStore()
const router = useRouter();
const toastStore = useToastStore()
const username_input = ref('')
const password_input = ref('')

const usernameEmpty = ref(false)
const passwordEmpty = ref(false)
const erroLogin = ref(false)


let btn = document.querySelector('#olho')
// Criar uma referência para o input
const inputPassword = ref(null);

function showPassword() {
    if (inputPassword.value.type == 'password') {
        inputPassword.value.type = 'text'
    } else {
        inputPassword.value.type = 'password'
    }
}


async function loginMetod() {
    erroLogin.value = false
    if (username_input.value && password_input.value) {

        usernameEmpty.value = false
        passwordEmpty.value = false

        const dataObject = {
            username: username_input.value,
            password: password_input.value
        }

        try {
            await login.loginAction(dataObject)
            if (login.token) {
                router.push({ name: 'home' }); // Redireciona para a rota 'home'
            }
        } catch (e) {
            if(e.status == "401"){
                erroLogin.value = true
            }else{
                toastStore.showToast(
                    e.status, e.data, 2
                )
            }
        }





    } else {
        if (!username_input.value) {
            usernameEmpty.value = true
        } else {
            usernameEmpty.value = false
        }
        if (!password_input.value) {
            passwordEmpty.value = true
        } else {
            passwordEmpty.value = false
        }
    }
}
const backgroundStyle = {
    backgroundImage: `url(${fundo_login})`,
    backgroundRepeat: "no-repeat",
    backgroundSize: `cover`
}
</script>
<template>

    <div class="row container-login g-0 ">
        <div class="col justify-content-center align-items-center d-none d-lg-flex " :style="backgroundStyle">
            <img style="
            width: 90%;
            " :src="logo" alt="">
        </div>
        <div class="col ">
            <section class="d-flex align-items-center h-100 container-de-fora">

                <div class="container-campos ">
                    <div class="container-tight">
                        <div class="d-flex align-items-center justify-content-between mb-5">
                            <div>
                                <h2 class="fs-1  bold mb-1 mt-0 p-0">Prazer em te ver!</h2>
                                <p class="fs-5 m-0 p-0"> Digite seu usuário e senha para logar</p>
                            </div>
                            <div>
                                <!-- Botão para mudar para o tema escuro -->
                                <button class="nav-link px-0 hide-theme-dark" title="Habilitar modo escuro"
                                    data-bs-toggle="tooltip" data-bs-placement="bottom" @click="theme.setTheme('dark')">
                                    <span v-html="svgs.darkThemeIcon"></span>
                                </button>

                                <!-- Botão para mudar para o tema claro -->
                                <button class="nav-link px-0 hide-theme-light" title="Habilitar modo claro"
                                    data-bs-toggle="tooltip" data-bs-placement="bottom"
                                    @click="theme.setTheme('light')">
                                    <span v-html="svgs.lightThemeIcon"></span>
                                </button>
                            </div>
                        </div>

                        <div class="card-body">


                            <form @submit.prevent="loginMetod" id="form" method="post" autocomplete="off" novalidate>

                                <div class="mb-3">
                                    <label class="form-label">Nome Usuário</label>
                                    <input v-model="username_input" id="username" type="text" class="formulario"
                                        :class="usernameEmpty ? 'is-invalid' : ''" name="username" placeholder="Digite um usuário"
                                       >
                                    <div v-if="usernameEmpty" class="invalid-feedback"> Usuário não
                                        digitado
                                    </div>
                                </div>
                                <div class="mb-2">
                                    <label class="form-label">
                                        Senha
                                    </label>
                                    <div class="">
                                        <input v-model="password_input" ref="inputPassword" type="password" 
                                            class="formulario" :class="passwordEmpty ? 'is-invalid' : ''"
                                            name="password" placeholder="Informe uma senha"  >
                                        <div v-if="passwordEmpty" class="invalid-feedback"> Senha não
                                            digitada
                                        </div>
                                    </div>
                                    <div class="mt-3 d-flex gap-2 align-items-center">
                                        <input type="checkbox" name="" id="mostrarsenha" @change="showPassword" >
                                        <label for="mostrarsenha" >Mostrar senha</label>
                                    </div>

 
                                </div>

                                <div v-if="erroLogin" class="form-footer">
                                    <span class="badge bg-red-lt w-100" style="height: 50px; padding:20px;">Usuario
                                        ou senha invalidos, tente novamente</span>
                                </div>

                                <div class="form-footer">
                                    <input type="submit" class="btn btn-primary w-100 rounded" value="Entrar">
                                </div>
                            </form>
                        </div>
                        <div class="mt-3 text-center">
                            <p>Não possui conta?   <router-link to="/sing-in"
                                    href="">criar conta</router-link></p>
                        </div>

                    </div>
                   
                </div>

            </section>
        </div>

    </div>


</template>
<style scoped>
.fundo-login {
    background-size: auto;
}


.formulario{
    min-width: 400px;
    padding: 10px 20px;
    border-radius: 30px;
    background: none;
    color: white;
    border-bottom: 2px solid #6f6e7457 ;
    border-top: 2px solid #5a2cff2a ;
    border-left: 2px solid #5a2cff2a ;
    border-right: 2px solid #5a2cff2a ;
    appearance: none;
    outline: none;
}
.formulario:focus{

border-bottom: 2px solid #4d41f0ab ;
outline: none;
box-shadow: none; 


}
[data-bs-theme=light] .formulario {
    color: black;
}
.container-campos{
    margin-left: 0;

    padding: 20px;
    border-radius: 20px;
}
.container-de-fora{
    justify-content: center;
}
@media (min-width: 992px) { /* Tamanho acima de md (768px) */
    .container-campos{
    margin-left: 70px;
    background: linear-gradient(to bottom, #ffffff00, #4b4a4a31) !important;
    padding: 20px;
    border-radius: 20px;
    
}
.container-de-fora{
    justify-content: start;

}
}
[data-bs-theme=dark] .container-campos {
    background: linear-gradient(to bottom, #14141400, #ffffff31) !important;
}
:root {
    --tblr-font-sans-serif: 'Inter Var', -apple-system, BlinkMacSystemFont, San Francisco, Segoe UI, Roboto, Helvetica Neue, sans-serif;
}

.body {
    font-feature-settings: "cv03", "cv04", "cv11";
}

.logo {
    max-width: 80%;
}



.alerta {
    height: 100vh;
    font-size: 40px;
    font-weight: bold;
}
</style>