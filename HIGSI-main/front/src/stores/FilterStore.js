import { defineStore } from 'pinia';

export const useFilterStore = defineStore('filter', {
  state: () => ({
    inputValue: ''
  }),
  actions: {
    setInputValue(value) {
        this.inputValue = value;  
      }
  }
});