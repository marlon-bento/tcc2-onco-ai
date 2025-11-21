import { defineStore } from 'pinia'


export const useLoginStore = defineStore('login', {
    
    state() {
        return {
            token: '',
            usuario: '',
        }
    },
    actions: {
        setRouter(router) {
            this.router = router;
        },
        loginVerific(){
            console.log('verificando login')
            this.token = localStorage.getItem('token') || ''
            this.usuario = localStorage.getItem('username') || ''
           
        },
        logoutAction(){
            console.log("fiz logout")
            this.token = '',
            this.usuario = '',
            localStorage.removeItem('token') 
            localStorage.removeItem('username')      
        },
        async loginAction(usuario) {
           const axios = this.$app.config.globalProperties.$api;
            try {
                const response = await axios.post('/user/login', {

                    username: usuario.username,
                    password: usuario.password,

                  }, {
                    headers: {
                      'Content-Type': 'application/x-www-form-urlencoded',
                      'accept': 'application/json'
                    }
                  })
                this.token = response.data.access_token
                localStorage.setItem('token', this.token)

                try {
                    const responseUser = await axios.get('/user/me',
                        {
                            headers: {
                                'Authorization': `Bearer ${response.data.access_token}`
                            }
                        })
                   
                    this.usuario = responseUser.data.user 
                    localStorage.setItem('token', this.token)
                    localStorage.setItem('username', this.usuario)
             } catch (e) {
                console.log(e)
                }
            }
            catch (e) {
                if (e.response) {
                    // Caso tenha resposta, mas com status de erro
                    const error = new Error('Erro ao obter os dados do usuário');
                    error.status = e.response.status;  // Status da resposta, por exemplo 401
                    error.data = e.response.data;  // Dados da resposta do servidor
                    throw error;
                }
                else if (e.request) {
                    const error = new Error('Erro ao obter os dados do usuário: Falha na requisição (servidor inacessível)');
                    error.status = 'No Response';
                    error.data = "Servidor não está respondendo";  // Não há dados se não houve resposta
                    throw error;
                }

            }


        },
        async criarConta(usuario) {
            const axios = this.$app.config.globalProperties.$api;
            try {
                const response = await axios.post('/user/create', {
                    email: usuario.email,
                    username: usuario.username,
                    password: usuario.password,

                  },)
                
            }
            catch (e) {
                if (e.response) {
                    // Caso tenha resposta, mas com status de erro
                    const error = new Error('Erro ao obter os dados do usuário');
                    error.status = e.response.status;  // Status da resposta, por exemplo 401
                    error.data = e.response.data;  // Dados da resposta do servidor
                    throw error;
                }
                else if (e.request) {
                    const error = new Error('Erro ao obter os dados do usuário: Falha na requisição (servidor inacessível)');
                    error.status = 'No Response';
                    error.data = "Servidor não está respondendo";  // Não há dados se não houve resposta
                    throw error;
                }

            }


        },
    },

})
