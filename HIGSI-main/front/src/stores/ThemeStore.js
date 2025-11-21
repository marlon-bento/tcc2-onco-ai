import {defineStore} from 'pinia'

export const useThemeStore = defineStore('theme', {
    //propriedades reativas
    state(){
        return{
           
        }
    },
    // metodos
    actions:{
        setTheme (theme) {
            // Altera o tema
            document.body.setAttribute('data-bs-theme', theme);
            localStorage.setItem('tablerTheme', theme); // Salva a preferência no local storage
            
            // Atualiza a URL sem recarregar a página
            //const url = new URL(window.location);
            //url.searchParams.set('theme', theme); // Adiciona ou atualiza o parâmetro 'theme'
            //window.history.pushState({}, '', url); // Atualiza a URL no navegador
        },
        getTheme(){
            let theme
            theme = document.body.getAttribute('data-bs-theme');
            return theme
        },
        verify(){
            const tabler_theme = localStorage.getItem('tablerTheme')
            if(tabler_theme){
                document.body.setAttribute('data-bs-theme', tabler_theme);
            }else{
                document.body.setAttribute('data-bs-theme', "dark");
                localStorage.setItem('tablerTheme', "dark");
            }
        }
    },

})