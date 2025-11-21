<template>
  <div class="modal modal-blur fade" id="confirma-status" tabindex="-1" style="display: none;" aria-hidden="true">
   <div class="modal-dialog modal-dialog-centered" role="document">
     <div class="modal-content">
       <div class="modal-header">
         <h5 v-if="props.laudoEnviado.status === 'Aguardando-resposta'" class="modal-title" >
           Alteração Status de Manutenção
 
         </h5>
         <h5 v-if="props.laudoEnviado.status === 'em-progresso'" class="modal-title" >
           Alteração Status de Devolução
 
         </h5>
         <button type="button" class="btn-close" data-bs-dismiss="modal" aria-label="Close"></button>
       </div>
       <div v-if="props.laudoEnviado.status === 'Aguardando-resposta'"  class="modal-body">
         Você tem a intenção de atualizar o status de <b>"Aguardando Resposta"</b> para <b>"Concluído"</b> a fim de encerrar o laudo?
       </div>
       <div v-if="props.laudoEnviado.status === 'em-progresso'"  class="modal-body">
         Você tem a intenção de atualizar o status de <b>"Em Progresso"</b> para <b>"Devolvido ao Patrimônio"</b> a fim de encerrar o laudo?
       </div>
       <div class="modal-footer">
         <button type="button" class="btn me-auto" data-bs-dismiss="modal">Não, cancelar</button>
         <button @click="$emit('encerrarLaudo', props.laudoEnviado)" type="button" id="confirma-dev" class="btn btn-primary" >Sim, encerrar</button>
       </div>
     </div>
   </div>
 </div>
 </template>
 <script setup>
 
 const props = defineProps([ 'laudoEnviado'])
 const emit = defineEmits(['encerrarLaudo'])
 
 </script>
