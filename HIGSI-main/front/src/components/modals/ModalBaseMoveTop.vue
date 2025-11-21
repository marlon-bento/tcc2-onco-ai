<script setup>
/**
 * Este é o componente base para modais que tem animação começada de baixo para cima
 * @example 
 *   
 * @param show prop passada para abrir e fechar o modal
 * 
 * @returns "sem retorno"
 */
const props = defineProps(['show'])

</script>

<template>
    <Transition>
            <div v-if="show" class="modal-mask">
                <div class="click " @click="$emit('close')">

                </div>
                <div class="modal-container d-block mx-auto ">
                    <slot name="content"></slot>
                </div>
            </div>
    </Transition>
</template>

<style scoped>

.modal-mask {
    position: fixed;
    z-index: 9998;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    background-color: rgba(0, 0, 0, 0.5);
    display: flex;
    backdrop-filter: blur(10px);
    
}

.click {
    position: fixed;
    z-index: 9998;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
}

.modal-container {
    z-index: 999999999;
    overflow-y: auto;
    transition: all 0.3s ease;
}

/* estado inicial de entrada */
.v-enter-from .modal-container {
    opacity: 0;
    transform: translateY(900px);
}

/* tempo de transição de entrada */
.v-enter-active {
    transition: all 0.6s ease-out;
}

.v-enter-active .modal-container {
    transition: all 0.6s ease-out;
}

/* estado inicial de entrada */
.v-leave-to .modal-container {
    opacity: 0;
    transform: translateY(900px);
}

/* tempo de transição de entrada */
.v-leave-active {
    transition: all 0.6s ease-out;
}

.v-leave-active .modal-container {
    transition: all 1s ease-out;
}
</style>