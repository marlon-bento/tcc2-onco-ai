// store.js
import { useToast } from "vue-toastification";
import MyToastComponent from '@/components/toasts/MyToastComponent.vue'

import { defineStore } from "pinia";

export const useToastStore = defineStore("toastStore", () => {
  const showToast = (title, body, type) => {
    const toast = useToast();
    const content = {
      component: MyToastComponent,
      props: {
        content: {
          title: title,
          body: body,
        },
      },
    };
    // 1 = success, 2 = error, 3 = info
    switch (type) {
      case 1:
        toast.success(content, {});
        break;
      case 2:
        toast.error(content, {
          timeout: 7000,
        });
        break;
      case 3:
        toast.info(content, {});
        break;
      default:
        console.info("Tipo de toast não encontrado");
        break;
    }
  };
  return {
    showToast,
  };
});
