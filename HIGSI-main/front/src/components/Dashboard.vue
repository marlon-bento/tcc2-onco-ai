<template>
  <label class="form-label text-white d-block mb-2">Envie uma imagem de mamografia:</label>
  <v-required name="imagem_experimento" :active-error="activeError">
    <ArrasteOuEnvie v-model="models.banner" textToDrag="Arraste uma imagem ou clique para enviar a imagem"
      textToDrop="Solte a imagem aqui!" subText="(JPG, JPEG, PNG, GIF)" @error="handleErrorImageArrastaOuEnvie" />
    <v-rule message="Por favor, envie uma imagem para o experimento" :error="() => !models.banner" />
    <v-rule message="A imagem deve ser do tipo jpg, jpeg ou png" :error="() => imageValid(models.banner)" />
  </v-required>
  <div class="d-flex gap-2">
    <div class="tipo-lesao-control mt-3 mb-4">
      <label for="tipo-lesao-select" class="form-label text-white d-block mb-2">Tipo de Lesão:</label>
      <select id="tipo-lesao-select" v-model="tipoLesao" class="form-select w-auto d-inline-block"
        aria-label="Selecionar tipo de lesão">
        <option value="CALC">Calcificação</option>
        <option value="MASS">Massa</option>
      </select>
    </div>
    <div class="superpixel-control mt-3 mb-4">
      <label for="superpixel-select" class="form-label text-white d-block mb-2">Número de Superpixels:</label>
      <select id="superpixel-select" v-model.number="numSuperpixels" class="form-select w-auto d-inline-block"
        aria-label="Selecionar número de superpixels">
        <option v-for="option in superpixelOptions" :key="option" :value="option">
          {{ option }}
        </option>
      </select>

    </div>


  </div>


  <button @click="enviarImagem" :disabled="isLoading">
    <span v-if="isLoading">Processando...</span>
    <span v-else>Enviar Imagem</span>
  </button>

  <div v-if="isLoading" class="text-center my-3">
    <p>Carregando e processando imagens...</p>
  </div>

  <div v-if="originalImageUrl" class="mt-4 resultados-experimento">
    <h2 class="mb-3">Resultados do Processamento</h2>

    <div v-if="diagnosisResult && !diagnosisResult.error" class="row gx-3 mb-4">
      <div class="col-12">
        <div class="card-usuario h-100 p-3 text-center"
          :class="{ 'border-danger border-2': diagnosisResult.prediction === 'Maligno', 'border-success border-2': diagnosisResult.prediction === 'Benigno' }">

          <h3 class="title-card-sensores mb-0">Diagnóstico da IA</h3>

          <h1 class="display-4 my-2"
            :class="{ 'text-danger': diagnosisResult.prediction === 'Maligno', 'text-success': diagnosisResult.prediction === 'Benigno' }">
            {{ diagnosisResult.prediction }}
          </h1>

          <div class="row">
            <div class="col-6">
              <span class="title-card-sensores">Confiança (Maligno)</span>
              <p class="fs-5 mb-0">{{ (parseFloat(diagnosisResult.confidence_malignant) * 100).toFixed(2) }}%</p>
            </div>
            <div class="col-6">
              <span class="title-card-sensores">Confiança (Benigno)</span>
              <p class="fs-5 mb-0">{{ (parseFloat(diagnosisResult.confidence_benign) * 100).toFixed(2) }}%</p>
            </div>
          </div>

        </div>
      </div>
    </div>

    <div class="row gx-3">

      <div class="col-md-4">
        <div class="card-usuario h-100">
          <div class="p-3">
            <h5 class="title-card-sensores">1. Imagem Original</h5>
            <img :src="originalImageUrl" class="img-fluid rounded" alt="Imagem Original do Experimento">
          </div>
        </div>
      </div>

      <div class="col-md-4">
        <div class="card-usuario h-100">
          <div class="p-3">
            <h5 class="title-card-sensores">2. Imagem Pré-processada</h5>
            <p class="small text-muted">(Redimensionada e CLAHE)</p> <img :src="preprocessedImageUrl"
              class="img-fluid rounded" alt="Imagem Pré-processada">
          </div>
        </div>
      </div>

      <div class="col-md-4">
        <div class="card-usuario h-100">
          <div class="p-3">
            <h5 class="title-card-sensores">3. Visualização de Superpixels ({{ numSuperpixels }} nós)</h5>
            <p class="small text-muted">(SLIC sobreposto na imagem)</p>
            <img :src="superpixelImageUrl" class="img-fluid rounded" alt="Imagem com Superpixels">
          </div>
        </div>
      </div>

    </div>
  </div>


  <div class="div-content mt-5">
  </div>
</template>
<script setup>
import { onMounted, ref, reactive } from 'vue';
import { useLoginStore } from "@/stores/LoginStore";
import { useToastStore } from "@/stores/useToastStore";
import ArrasteOuEnvie from "@/components/ArrasteOuEnvie.vue";
import { initVrequired, VRequired, VRule } from "v-required/validation";

import { useApi } from "v-api-fetch";
const api = useApi();
const { haveError } = initVrequired()
const activeError = ref(false);
const models = reactive({
  banner: null,
});
const toast = useToastStore();
const login = useLoginStore();

const originalImageUrl = ref(null);
const preprocessedImageUrl = ref(null);
const superpixelImageUrl = ref(null);
const isLoading = ref(false);
const numSuperpixels = ref(50);
const tipoLesao = ref('MASS');
const superpixelOptions = [25, 50, 100, 200];

const diagnosisResult = ref(null);

function handleErrorImageArrastaOuEnvie(errorCode, message) {
  toast.showToast("Erro", `Erro no upload (${errorCode}): ${message}`, 2);
}

function imageValid(image) {
  if (!image || !image.name) return false;
  const validExtensions = ["jpg", "jpeg", "png"];
  const extension = image.name.split(".").pop().toLowerCase();

  return !validExtensions.includes(extension);
}

async function enviarImagem() {
  activeError.value = true
  if (haveError()) {
    toast.showToast("Erro", `Por favor, corrija os erros antes de enviar.`, 2);
    return;
  }
  if (!models.banner) {
    toast.showToast("Erro", "Nenhuma imagem selecionada para envio.", 2);
    return;
  }

  isLoading.value = true;
  originalImageUrl.value = null;
  preprocessedImageUrl.value = null;
  superpixelImageUrl.value = null;
  diagnosisResult.value = null;

  const formData = new FormData();
  formData.append('image', models.banner);
  formData.append('n_superpixels', numSuperpixels.value);
  formData.append('tipo_lesao', tipoLesao.value);

  toast.showToast("Info", "Enviando imagem para processamento...", 3);

  try {

    const response = await api.post('upload-experimento/', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });

    isLoading.value = false;

    if (response.data) {
      toast.showToast("Sucesso", "Diagnóstico recebido!", 1);

      originalImageUrl.value = response.data.original_image_url;
      preprocessedImageUrl.value = response.data.preprocessada_image_url;
      superpixelImageUrl.value = response.data.superpixel_image_url;

      diagnosisResult.value = response.data.diagnosis;

      models.banner = null;
      activeError.value = false;
    } else {
      toast.showToast("Aviso", "Imagem enviada, mas sem resposta de processamento.", 3);
    }

  } catch (error) {
    isLoading.value = false;
    console.error("Erro ao enviar imagem:", error);
    if (error.response && error.response.data) {
      let errorMessage = "Erro desconhecido ao enviar imagem.";
      if (error.response.data.detail) {
        errorMessage = error.response.data.detail;
      } else if (error.response.data.error) {
        errorMessage = error.response.data.error;
      } else if (error.response.data.image) {
        errorMessage = `Erro na imagem: ${error.response.data.image.join(', ')}`;
      } else {
        errorMessage = JSON.stringify(error.response.data);
      }
      toast.showToast("Erro", `Falha no envio: ${errorMessage}`, 2);
    } else {
      toast.showToast("Erro", `Falha na comunicação com o servidor: ${error.message}`, 2);
    }
  }
}
onMounted(() => {
  login.loginVerific();
});
</script>

<style scoped>
.div-content {
  margin-bottom: 20px;


}

.card-temperatura {
  width: 230px;
  height: 70px;
  border-radius: 10px;
  background: linear-gradient(60deg, #22232b, #1a1f37b2) !important;
}

.card-sensores {
  width: 300px;
  height: 70px;
  border-radius: 10px;
  background: linear-gradient(60deg, #060b26, #1a1f37b2) !important;
}

.card-usuario {
  border-radius: 20px;
  background: linear-gradient(60deg, #060b26, #1a1f37b2) !important;
}

.title-card-sensores {
  color: #A0AEC0;
  font-size: 14px;
}

body {
  display: flex;
  justify-content: center;
  align-items: center;
  height: 100vh;
  margin: 0;
  background: #1e1e2d;
  color: white;
  font-family: Arial, sans-serif;
}

.gauge {
  position: relative;
  width: 200px;
  height: 200px;
  display: flex;
  justify-content: center;
  align-items: center;

}

.gauge svg {
  position: absolute;
  transform: rotate(-230deg);
  /* Roda o gráfico para que o preenchimento comece de cima */
}

.gauge2 {
  position: relative;
  width: 200px;
  height: 200px;
  display: flex;
  justify-content: center;
  align-items: center;

}

.gauge2 svg {
  position: absolute;
  transform: rotate(40deg);
  /* Roda o gráfico para que o preenchimento comece de cima */
}

.content-dados-co {
  background: #111946;
  width: 80%;
  padding: 10px 40px;
  border-radius: 20px;
  margin-bottom: 20px;
}


#gauge-value {
  font-size: 1.5rem;
  font-weight: bold;
}

#gauge-value2 {
  font-size: 1.5rem;
  font-weight: bold;
}

#gauge-status {
  font-size: 1rem;
}
</style>
