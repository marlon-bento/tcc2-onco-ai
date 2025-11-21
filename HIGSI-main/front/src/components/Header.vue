<script setup>
import { useLoginStore } from "@/stores/LoginStore";
import { useThemeStore } from "@/stores/ThemeStore";
import img_logo from '@/assets/static/imgs/logo.png'
import { useRouter } from 'vue-router';
import { ref, watch, onMounted } from "vue";
import CryptoJS from 'crypto-js';
import defaultGravatar from "@/assets/static/imgs/avatar.svg";
import { IconLayoutDistributeVertical } from "@tabler/icons-vue";
import cleanAir from "@/assets/static/imgs/texto-onco.png"

//import { useLoginStore, useThemeStore } from '@/stores';
// Variável reativa para o tema
const login = useLoginStore()
const theme = useThemeStore()
const gravatarUrl = ref('');
const router = useRouter();


function filter_meus() {
    const filtro = login.last_name;
    const inputElement = document.getElementById('inputSearchLaudos');
    const event = new Event('input', { bubbles: true });
    inputElement.value = filtro;
    inputElement.dispatchEvent(event);

};

function filter_progresso() {
    const filtro = "em-progresso";
    const inputElement = document.getElementById('inputSearchLaudos');
    const event = new Event('input', { bubbles: true });
    inputElement.value = filtro;
    inputElement.dispatchEvent(event);
};

function filter_devolvido() {
    const filtro = "Devolvido-ao-patrimonio";
    const inputElement = document.getElementById('inputSearchLaudos');
    const event = new Event('input', { bubbles: true });
    inputElement.value = filtro;
    inputElement.dispatchEvent(event);
};

function filter_aguardando() {
    const filtro = "Aguardando-resposta";
    const inputElement = document.getElementById('inputSearchLaudos');
    const event = new Event('input', { bubbles: true });
    inputElement.value = filtro;
    inputElement.dispatchEvent(event);
};

function filter_concluido() {
    const filtro = "Concluido";
    const inputElement = document.getElementById('inputSearchLaudos');
    const event = new Event('input', { bubbles: true });
    inputElement.value = filtro;
    inputElement.dispatchEvent(event);
};

function filter_registrado() {
    const filtro = "Registrado";
    const inputElement = document.getElementById('inputSearchLaudos');
    const event = new Event('input', { bubbles: true });
    inputElement.value = filtro;
    inputElement.dispatchEvent(event);
};

// Função para gerar a URL do Gravatar
function generateGravatarUrl(email) {
    const hashedEmail = CryptoJS.SHA256(email.trim().toLowerCase()).toString();
    return `https://www.gravatar.com/avatar/${hashedEmail}?d=404`;
}
function handleError() {
    gravatarUrl.value = defaultGravatar; // Define a imagem padrão ao ocorrer erro
}
// Observa mudanças no e-mail no loginStore para atualizar o Gravatar
watch(
    () => login.email,
    (newEmail) => {
        if (newEmail) {
            gravatarUrl.value = generateGravatarUrl(newEmail);
        }
    }
);
onMounted(() => {
    if (login.email) {
        gravatarUrl.value = generateGravatarUrl(login.email);
    }
})



</script>

<template>
    <header class="container-fora  ">
        <div class="container-header">
            <div class="d-flex justify-content-center w-100 pt-5 logo elemento">
                <RouterLink class="d-flex align-items-center" to="/"><img :src="cleanAir" alt=""></RouterLink>
                <div class=""></div>
            </div>
            <ul class="mt-5 pt-3">
                <li>
                    <RouterLink to="/dashboard" class="dropdown-item">
                        <span class="nav-link-icon d-md-none d-lg-inline-block" v-html="svgs.menuDevolucaoIcon">

                        </span>
                        <span class="fs-1">
                            Dashboard
                        </span>
                    </RouterLink>

                </li>
                <li>
                    <RouterLink to="/" class="dropdown-item">
                        <span class="nav-link-icon d-md-none d-lg-inline-block" v-html="svgs.menuDevolucaoIcon">

                        </span>
                        <span class="fs-1">
                            Histórico
                        </span>
                    </RouterLink>
                </li>
                <li>

                </li>
                <li>

                </li>
                <li>

                </li>
            </ul>
        </div>

    </header>

</template>
<style scoped>
.elemento {
    position: relative;
    padding-bottom: 10px;
    /* Ajuste conforme necessário */
}

.elemento::after {
    content: '';
    position: absolute;
    bottom: -20px;


    width: 80%;
    height: 2px;
    /* Espessura da borda */
    background: linear-gradient(to right, rgba(0, 0, 0, 0.26) 0%, rgba(0, 0, 0, 0.397) 10%, rgba(255, 255, 255, 0.548) 40%, rgba(255, 255, 255, 0.534) 70%, rgba(0, 0, 0, 0.445) 80%, rgba(0, 0, 0, 0) 100%);

}

.container-fora {
    padding-top: 20px;
    padding-left: 20px;

}

.container-header {
    height: 100%;
    background: linear-gradient(120deg, #060b26e5 30%, #1a1f37 70%, #1a1f3700 100%) !important;
    border-radius: 20px 20px 0 0;

}

ul {
    list-style: none;
}



header {
    position: sticky;
    top: 0;
    /* Fixa o Header ao topo da coluna */
    width: 100%;
    /* Garante que ocupe apenas a largura da coluna */
    height: 100vh;
}
</style>