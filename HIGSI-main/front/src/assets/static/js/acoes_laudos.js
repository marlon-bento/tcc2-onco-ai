// Função que aciona os butões e é chamado no initcomplete no datatables
function acoes() {

    
            
    $('#table-aux').on('click', '.btnDetalhe' ,function() {
        const id = $(this).attr('id')
        const tipoLaudo = $(this).closest('tr').find('td:eq(4)').text();
        visualizar_laudos(id, tipoLaudo)
    })

    $('#table-aux').on('click', '.btnEdit', function() {
        const id = $(this).attr('id')
        const tipoLaudo = $(this).closest('tr').find('td:eq(4)').text()
        edit_laudos(id, tipoLaudo)
    })

    $('#table-aux').on('click', '.btnPrint', function () {
        const id = $(this).attr('id')
        const tipoLaudo = $(this).closest('tr').find('td:eq(4)').text();
        imprime_laudos(id, tipoLaudo)
    })

    $('#table-aux').on('click', '.btnStatus', function() {
        const id = $(this).attr('id')
        const tipoLaudo = $(this).closest('tr').find('td:eq(4)').text()
        altera_status(id, tipoLaudo)
    })
    

    $('.check-todos').click(function() {
        $('.checkbox').prop('checked', this.checked)
        $('#btn-imprime-checks').prop('disabled', ! this.checked); 
    })
    
    
    $('#table-aux').on('click', '.checkbox', function(e) {
        $('#btn-imprime-checks').prop('disabled', $('.checkbox:checked').length == 0);
        $('#check-todos').prop('checked', $('.checkbox:checked').length == $('.checkbox').length)
    }) 


}

// Esse é o techo de codigo de visualização dos laudos, onde primeiramente se compara qual laudo sera alterado, depois uma API que retorna os
// dados do laudo sera enviado para o front-end na requisição get, logo apos sera rederizado um modal (arquivo: modal-man/modal-dev) com a 
// alteração dos inputs para os valores do banco
function visualizar_laudos(id, tipolaudo) {

    
        $.get(`api-laudo-visu/${id}/`, (res) => {
            let dados = res.data
            
            if (tipolaudo == "Devolução"){
                
                $('#modal-nome-dev').val(dados.nome_solicitante)
                $('#modal-email-dev').val(dados.email_solicitante)
                $('#modal-unidade-dev').val(dados.unidade)
                $('#modal-instituto-dev').val(dados.instituto)
                $('#modal-departamento-dev').val(dados.setor)
                $('#modal-numchamado-dev').val(dados.num_chamado)
                $('#modal-ramal-dev').val(dados.num_ramal)
                $('#modal-predio-dev').val(dados.num_predio)
                $('#modal-sala-dev').val(dados.num_sala)
                $('#modal-tipoequip-dev').val(dados.tipo_equip)
                $('#modal-numserie-dev').val(dados.num_serie)
                $('#modal-patrimonio-dev').val(dados.patrimonio)
                $('#modal-marca-dev').val(dados.marca)
                $('#modal-modelo-dev').val(dados.modelo)
                $('#modal-observacoes-dev').val(dados.observacoes)

                $('#modal-devolucao').modal('show')
            } else {
                $('#modal-nome-man').val(dados.nome_solicitante)
                $('#modal-email-man').val(dados.email_solicitante)
                $('#modal-unidade-man').val(dados.unidade)
                $('#modal-instituto-man').val(dados.instituto)
                $('#modal-departamento-man').val(dados.setor)
                $('#modal-numchamado-man').val(dados.num_chamado)
                $('#modal-ramal-man').val(dados.num_ramal)
                $('#modal-predio-man').val(dados.num_predio)
                $('#modal-sala-man').val(dados.num_sala)
                $('#modal-tipoequip-man').val(dados.tipo_equip)
                $('#modal-numserie-man').val(dados.num_serie)
                $('#modal-patrimonio-man').val(dados.patrimonio)
                $('#modal-marca-man').val(dados.marca)
                $('#modal-modelo-man').val(dados.modelo)
                $('#modal-cc-man').val(`${dados.nome_centro_custo}: ${dados.centro_custo}.${dados.unidade_negocio}`)
                $('#modal-observacoes-man').val(dados.observacoes)

                $('#modal-manutencao').modal('show')
            }
        })
        
         
    
} 

// Esse é o trecho de codigo da edição de laudo, onde primeiramente se compara qual laudo sera alterado, entre manutenção e manolução e 
// depois chama na view a função de edição do laudo selecionado
function edit_laudos(id, tipolaudo) {
    if (tipolaudo == 'Manutenção') {
        window.location.href = `edit-man/${id}/`
    } else {
        window.location.href = `edit-dev/${id}/`
    }
}



// Esse trecho de codigo chama o modal da confirmação da alteração de status, para o controle dos laudos
function altera_status(id, tipo_laudo){ 
    if (tipo_laudo == "Manutenção"){
        $('#confirma-status-man').modal('show')
        
        $('#confirma-man').off('click').on('click', function() {
            $.ajax({
                url: `api_altera_status/${id}/`,
                type: 'POST',
                success: function(res) {


                    $(`.aguardando_resp[id="${id}"]`).replaceWith(`<span title="Concluído" class="badge bg-green-lt" 
                    data-bs-toggle="tooltip" data-bs-placement="top">Concluído</span>`)

                    $(`.btnStatus[id="${id}"]`).replaceWith(`<svg title="concluido" xmlns="http://www.w3.org/2000/svg" width="24" 
                    height="24" viewBox="0 0 24 24" stroke-width="2" stroke="currentColor" fill="none" 
                    stroke-linecap="round" stroke-linejoin="round" data-bs-toggle="tooltip" data-bs-placement="top">
                        <path stroke="none" d="M0 0h24v24H0z" fill="none" ></path>
                        <path d="M7 12l5 5l10 -10"></path>
                        <path d="M2 12l5 5m5 -5l5 -5"></path>
                    </svg>`)
                    
                    let myToast = $.toast({
                        heading: 'Adicionado!',
                        text: 'O status foi alterado com sucesso!',
                        showHideTransition: 'slide',
                        icon: 'success',
                        position: 'top-center'
                    })
                   

                },
                error: function(err){
                    alert('deu ruim na atualização')
                }
            })
        })

    } else {
        $('#confirma-status-dev').modal('show')

        $('#confirma-dev').off('click').on('click', function() {
            $.ajax({
                url: `api_altera_status/${id}/`,
                type: 'POST',
                success: function(res) {


                    $(`.em-progresso[id="${id}"]`).replaceWith(`<span title="Devolvido ao patrimônio" class="badge bg-green-lt ms-auto"
                    data-bs-toggle="tooltip" data-bs-placement="top">Encerrado</span>`)

                    $(`.btnStatus[id="${id}"]`).replaceWith(`<svg title="Encerrado" xmlns="http://www.w3.org/2000/svg" width="24" 
                    height="24" viewBox="0 0 24 24" stroke-width="2" stroke="currentColor" fill="none" 
                    stroke-linecap="round" stroke-linejoin="round" data-bs-toggle="tooltip" data-bs-placement="top">
                        <path stroke="none" d="M0 0h24v24H0z" fill="none" ></path>
                        <path d="M7 12l5 5l10 -10"></path>
                        <path d="M2 12l5 5m5 -5l5 -5"></path>
                    </svg>`)
                    
                    let myToast = $.toast({
                        heading: 'Adicionado!',
                        text: 'O status foi alterado com sucesso!',
                        showHideTransition: 'slide',
                        icon: 'success',
                        position: 'top-center'
                    })
                   

                },
                error: function(err){
                    alert('deu ruim na atualização')
                }
            })
        })

    }
}


