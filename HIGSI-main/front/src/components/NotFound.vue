<template>
    <div class=" erro d-flex flex-column gap-3 justify-content-center align-items-center">
        <p class="bold">Opps.. Sinto muito, Página não encontrada!!!</p>
        <img :src="personagem_404" alt="">
        <RouterLink class="btn btn-info" to="/" ><span class="bold fs-1">Voltar</span></RouterLink>
    </div>
   
</template>
<script setup>
import personagem_404 from '@/assets/static/imgs/personagem404.png'
</script>
<style scoped>
.erro{
 height: 100vh;
 width: 100%;
}
.erro img{
    width: 370px;
}
.erro p{
    font-size: 30px;
}
</style>