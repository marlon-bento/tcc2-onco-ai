<script setup>
/**
 * Este é um componente usado para passar o coteudo do modal para ModalBaseMoveTop.vue
 * os dados são passados via slot para o ModalBase
 
 * @param   showprop passada para abrir e fechar o modal
 * @param   laudo "recebe todos os dados do laudo inclusive o tipo para saber como exibir o modal"
 * @returns "sem retorno"
 */
const props = defineProps(['show', 'laudo']);
import ModalBaseMoveTop from '@/components/modals/ModalBaseMoveTop.vue';


</script>

<template>
  <ModalBaseMoveTop :show="props.show" @close="$emit('close')">
    <template #content>
      <div class=" modal-container ">
        <div class="" role="document">
          <div class="d-flex flex-column">
            <div class="header-modal py-3 px-4">
              <h5 v-if="laudo.tipo_laudo === 'devolucao'" class="modal-title ">Visualização Laudo Devolução</h5>
              <h5 v-if="laudo.tipo_laudo === 'tecnico'" class="modal-title ">Visualização Laudo Técnico</h5>
              <h5 v-if="laudo.tipo_laudo === 'manutencao'" class="modal-title ">Visualização Laudo Manutenção</h5>
              
              <button @click="$emit('close')" type="button" class="button-close" >
                <span class=" inputClose" v-html="svgs.inputClose"></span>
              </button>
            </div>
            <div class="modal-body py-3 px-4">
              <div class="form-selectgroup-boxes row mb-3">
                <div class="col-lg-6">
                  <div class="mb-3">
                    <label class="form-label">Nome Solicitante</label>
                    <input type="text" id="modal-nome-dev" class="form-control" name="example-text-input"
                      placeholder="Nome Solicitante" :value="props.laudo.nome_solicitante" readonly>
                  </div>
                </div>
                <div class="col-lg-6">
                  <div class="mb-3">
                    <label class="form-label">Email Solicitante</label>
                    <input type="text" id="modal-email-dev" class="form-control" name="example-text-input"
                      placeholder="Email Solicitante" :value="props.laudo.email_solicitante" readonly>
                  </div>
                </div>
              </div>
              <div class="form-selectgroup-boxes row mb-3">
                <div class="col-lg-6">
                  <div class="mb-3">
                    <label class="form-label">Unidade</label>
                    <input type="text" id="modal-unidade-dev" class="form-control" name="example-text-input"
                      placeholder="Unidade" :value="props.laudo.nome_unidade" readonly>
                  </div>
                </div>
                <div class="col-lg-6">
                  <div class="mb-3">
                    <label class="form-label">Instituto</label>
                    <input type="text" id="modal-instituto-dev" class="form-control" name="example-text-input"
                      placeholder="Instituto" :value="props.laudo.instituto" readonly>
                  </div>
                </div>
                <div class="col-lg-6">
                  <div class="mb-3">
                    <label class="form-label">Setor</label>
                    <input type="text" id="modal-departamento-dev" class="form-control" name="example-text-input"
                      placeholder="Departamento" :value="props.laudo.setor" readonly>
                  </div>
                </div>

                <div class="col-lg-6">
                  <div class="mb-3">
                    <label class="form-label">Número do CSC</label>
                    <input type="text" id="modal-numchamado-dev" class="form-control" name="example-text-input"
                      placeholder="Número do Chamado" :value="props.laudo.num_chamado" readonly>
                  </div>
                </div>
              </div>
              <div class="form-selectgroup-boxes row mb-3">
                <div class="col-lg-4">
                  <div class="mb-3">
                    <label class="form-label">Número do Ramal</label>
                    <input type="text" id="modal-ramal-dev" class="form-control" name="example-text-input"
                      placeholder="Número do Ramal" :value="props.laudo.num_ramal" readonly>
                  </div>
                </div>
                <div class="col-lg-4">
                  <div class="mb-3">
                    <label class="form-label">Número do Prédio</label>
                    <input type="text" id="modal-predio-dev" class="form-control" name="example-text-input"
                      placeholder="Número do Prédio" :value="props.laudo.num_predio" readonly>
                  </div>
                </div>
                <div class="col-lg-4">
                  <div class="mb-3">
                    <label class="form-label">Número da Sala</label>
                    <input type="text" id="modal-sala-dev" class="form-control" name="example-text-input"
                      placeholder="Número da Sala" :value="props.laudo.num_sala" readonly>
                  </div>
                </div>
              </div>
              <h5 class="modal-title">Dados do Equipamento</h5>
              <div class="col-lg-12">
                <div class="mb-3">
                  <label class="form-label">Tipo de Equipamento</label>
                  <input type="text" id="modal-tipoequip-dev" class="form-control" name="example-text-input"
                    placeholder="Tipo do Equipamento" :value="props.laudo.tipo_equip" readonly>
                </div>
              </div>
              <div class="form-selectgroup-boxes row mb-3">
                <div class="col-lg-6">
                  <div class="mb-3">
                    <label class="form-label">Número de Série</label>
                    <input type="text" id="modal-numserie-dev" class="form-control" name="example-text-input"
                      placeholder="Número de Série" :value="props.laudo.num_serie" readonly>
                  </div>
                </div>
                <div class="col-lg-6">
                  <div class="mb-3">
                    <label class="form-label">Número do Patrimônio</label>
                    <input type="text" id="modal-patrimonio-dev" class="form-control" name="example-text-input"
                      placeholder="Número do Patrimônio" :value="props.laudo.patrimonio" readonly>
                  </div>
                </div>
              </div>
              <div class="form-selectgroup-boxes row mb-3">
                <div class="col-lg-6">
                  <div class="mb-3">
                    <label class="form-label">Marca do Equipamento</label>
                    <input type="text" id="modal-marca-dev" class="form-control" name="example-text-input"
                      placeholder="Marca do Equipamento" :value="props.laudo.marca" readonly>
                  </div>
                </div>
                <div class="col-lg-6">
                  <div class="mb-3">
                    <label class="form-label">Modelo do Equipameto</label>
                    <input type="text" id="modal-modelo-dev" class="form-control" name="example-text-input"
                      placeholder="Modelo do Equipamento" :value="props.laudo.modelo" readonly>
                  </div>
                </div>
              </div>


              <div v-if="laudo.tipo_laudo === 'manutencao'" class="mb-3">
                <label class="form-label">Centro de Custo</label>
                <input type="text" id="modal-cc-man" class="form-control" name="example-text-input" placeholder="Centro de Custo" :value="`${props.laudo.nome_centro_custo}:  ${props.laudo.centro_custo}.${props.laudo.unid_negocio}`" readonly="">
              </div>


              
              <div class="col-lg-12">
                <div class="mb-3">
                  <label class="form-label">Observações e testes realizados</label>
                  <textarea type="text" id="modal-observacoes-dev" style="height: 90px;" class="form-control"
                    name="example-text-input" placeholder="Observações e testes realizados" :value="props.laudo.observacoes" readonly></textarea>
                </div>
              </div>
              
              <div class="modal-footer">
                <a @click.prevent="$emit('close')" class="btn btn-danger" >
                  Voltar
                </a>
              </div>
            </div>
          </div>
        </div>
      </div>
    </template>
  </ModalBaseMoveTop>
</template>

<style scoped>
.modal-container{
  background-color: var(--tblr-bg-surface) !important;
  margin: 20px 0px;
  border-radius: 7px;

}
.header-modal{
  border-bottom: 1px solid rgba(187, 186, 186, 0.37) !important;
  display: flex;
  justify-content: space-between;
}
.button-close{
  border: none;
  background: none;
}
</style>