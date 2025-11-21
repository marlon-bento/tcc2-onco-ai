<template>
    
    <div class="image-drop-zone form-control" :class="{ 'is-dragging': internalIsDraggingOver }"
        @dragenter.prevent="handleDragEnter" @dragover.prevent="handleDragOver" @dragleave.prevent="handleDragLeave"
        @drop.prevent="handleDrop" @click="triggerFileInput">
        <input type="file" ref="actualFileInputRef" @change="handleFileSelectedFromInput"
            :accept="props.acceptedMimeTypes.join(',')" style="display: none;" />

        <div v-if="!currentDisplayUrl && !displayOldImageUrl" class="uploader-prompt">
            <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none"
                stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"
                class="feather feather-upload-cloud">
                <polyline points="16 16 12 12 8 16"></polyline>
                <line x1="12" y1="12" x2="12" y2="21"></line>
                <path d="M20.39 18.39A5 5 0 0 0 18 9h-1.26A8 8 0 1 0 3 16.3"></path>
                <polyline points="16 16 12 12 8 16"></polyline>
            </svg>
            <p v-if="internalIsDraggingOver" class="text-drop">{{ props.textToDrop }}</p>
            <p v-else>{{ props.textToDrag }}</p>
            <small>{{ props.subText }}</small>
        </div>
        <div v-if="currentDisplayUrl" class="image-preview-container">
            <img :src="currentDisplayUrl" alt="Pré-visualização da imagem" class="image-preview" />
            <button type="button" @click.stop="removeImage" class="btn btn-sm btn-danger remove-image-button">
                Remover
            </button>
        </div>
        <div v-else-if="displayOldImageUrl" class="image-preview-container">
            <div class="bg-preview">
                <span>Imagem atual</span>
            </div>
            <img :src="displayOldImageUrl" alt="Pré-visualização da imagem" class="image-preview" />
        </div>
    </div>
</template>

<script setup>
import { ref, onUnmounted, watch, computed } from "vue";


const props = defineProps({
    modelValue: { // Para v-model do File object
        type: [File, null, String],
        default: null
    },
    textToDrag: {
        type: String,
        default: "Arraste uma imagem ou clique para selecionar"
    },
    textToDrop: {
        type: String,
        default: "Solte a imagem aqui!"
    },
    subText: {
        type: String,
        default: "(JPG, PNG, GIF)"
    },
    acceptedMimeTypes: {
        type: Array,
        default: () => ['image/jpeg', 'image/png', 'image/gif', 'image/jpg']
    },
    initialDisplayImageUrl: { // Para exibir uma imagem já existente (URL estática)
        type: String,
        default: null
    }
});

const emit = defineEmits([
    'update:modelValue', // Para v-model do File
    'error' // Para erros de validação
]);

// Refs internas do componente
const actualFileInputRef = ref(null);
const internalImagePreviewUrl = ref(null); // URL de preview gerado localmente (blob)
const internalIsDraggingOver = ref(false);

// URL final a ser exibida: prioriza o preview local, depois a URL inicial
const currentDisplayUrl = computed(() => {
    return internalImagePreviewUrl.value;
});

const displayOldImageUrl = computed(() =>{
    return props.initialDisplayImageUrl
})

// Observa mudanças no modelValue (arquivo) vindo do pai
watch(() => props.modelValue, (newFile) => {
    if (newFile && newFile instanceof File) {
        if (!internalImagePreviewUrl.value || (internalImagePreviewUrl.value && !internalImagePreviewUrl.value.startsWith('blob:'))) {
            // Gera preview se não houver um blob local ou se o arquivo for diferente
            generatePreview(newFile);
        }
    } else if (!newFile) {
        // Se o pai limpar o arquivo (modelValue = null)
        cleanupLocalPreviewAndFileInput();
    }
});

// Observa mudanças na URL de imagem inicial vinda do pai
watch(() => props.initialDisplayImageUrl, (newUrl) => {
    if (newUrl && !props.modelValue && !internalImagePreviewUrl.value) {
        // Se uma URL inicial for fornecida e não houver arquivo local/preview
        // Não precisamos fazer nada aqui, o computed 'currentDisplayUrl' já vai pegar
    } else if (!newUrl && !props.modelValue) {
        cleanupLocalPreviewAndFileInput();
    }
});


const cleanupLocalBlobUrl = () => {
    if (internalImagePreviewUrl.value && internalImagePreviewUrl.value.startsWith('blob:')) {
        URL.revokeObjectURL(internalImagePreviewUrl.value);
        internalImagePreviewUrl.value = null;
    }
};

const cleanupLocalPreviewAndFileInput = () => {
    cleanupLocalBlobUrl();
    internalImagePreviewUrl.value = null; // Garante que o preview local seja limpo
    if (actualFileInputRef.value) {
        actualFileInputRef.value.value = ''; // Limpa o valor do input de arquivo
    }
};

onUnmounted(() => {
    cleanupLocalBlobUrl();
});

const generatePreview = (file) => {
    cleanupLocalBlobUrl(); // Limpa o anterior, se houver
    internalImagePreviewUrl.value = URL.createObjectURL(file);
};

const processFile = (file) => {
    if (!file) return;

    if (!props.acceptedMimeTypes.includes(file.type)) {
        
        const friendlyAcceptedTypes = props.acceptedMimeTypes.map(mimeType => {
        if (typeof mimeType === 'string' && mimeType.includes('/')) {
            return mimeType.split('/')[1].toUpperCase(); // Pega 'JPG' de 'image/jpg'
        }
        // Fallback caso o tipo MIME não siga o padrão esperado
        return typeof mimeType === 'string' ? mimeType.toUpperCase() : '';
        }).filter(type => type !== '').join(', '); // Filtra vazios e junta com ', '
        

        const errorMsg = `Tipo de arquivo inválido (${file.type.split('/')[1].toUpperCase()}). Use com extensão ${friendlyAcceptedTypes}.`;
        emit('error', 'tipo_invalido', errorMsg); // Emite erro para o pai
        cleanupLocalPreviewAndFileInput(); // Limpa o input
        emit('update:modelValue', null); // Garante que o modelValue inválido seja limpo
        return;
    }

    generatePreview(file);
    emit('update:modelValue', file); // Emite o objeto File para o v-model
};

const handleDragEnter = () => {
    internalIsDraggingOver.value = true;
};

const handleDragOver = () => {
    internalIsDraggingOver.value = true;
};

const handleDragLeave = (event) => {
    if (event.currentTarget.contains(event.relatedTarget)) return;
    internalIsDraggingOver.value = false;
};

const handleDrop = (event) => {
    internalIsDraggingOver.value = false;
    const files = event.dataTransfer.files;
    if (files && files.length > 0) {
        processFile(files[0]);
    }
};

const triggerFileInput = () => {
    if (!currentDisplayUrl.value && actualFileInputRef.value) { // Só aciona se não houver imagem
        actualFileInputRef.value.click();
    }
};

const handleFileSelectedFromInput = (event) => {
    const files = event.target.files;
    if (files && files.length > 0) {
        processFile(files[0]);
    }
};

const removeImage = () => {
    cleanupLocalPreviewAndFileInput();
    emit('update:modelValue', null); // Emite null para o v-model
};

// Se o componente for carregado com um modelValue (File) já definido, gera o preview
if (props.modelValue instanceof File && !internalImagePreviewUrl.value) {
    generatePreview(props.modelValue);
}

</script>

<style scoped>

.image-drop-zone {
    border: 2px dashed #ccc;
    padding: 20px;
    text-align: center;
    cursor: pointer;
    transition: background-color 0.2s, border-color 0.2s;
    min-height: 150px;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    position: relative;
}

.image-drop-zone.is-dragging {
    border-color: #007bff !important; 
    background-color: #f0f8ff;
}

.uploader-prompt svg {
    margin-bottom: 10px;
    color: #6c757d;
}

.uploader-prompt p {
    margin-bottom: 5px;
    font-size: 1rem;
}

.uploader-prompt small {
    font-size: 0.8rem;
    color: #6c757d;
}

.image-preview-container {
    position: relative;
    display: inline-block;
}

.image-preview {
    max-width: 100%;
    max-height: 200px;
    display: block;
    margin-bottom: 10px;
}

.remove-image-button {
    position: absolute;
    top: 5px;
    right: 5px;
    z-index: 10;
}
.bg-preview{
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    background-color: rgba(0, 0, 0, 0.5);
    display: flex;
    align-items: center;
    justify-content: center;
    z-index: 40;
    font-size: 30px;
    font-weight: bold;

}
.bg-preview span{
    background-color: rgba(0, 0, 0, 0.821);
    padding: 5px 10px;
}
.text-drop{
    color: black;
    font-weight: bold;
}
</style>