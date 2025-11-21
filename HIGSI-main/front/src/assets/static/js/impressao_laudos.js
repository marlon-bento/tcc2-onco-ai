// Função desse .js é imprimir os laudos, na view existem duas APIs que me retornam os dados de cada um dos laudos, é feita a comparação de 
// qual esta sendo requerido, e então os dados são retornados, depois disso acontece o draw da pagina de impressão no content, tambem à 
// algumas adaptações para caber em uma folha a4, como exemplo o unidade_split -> essa variavel corta parte do nome da unidade para não ficar
// tão grande. Função de click chamada em -> laudos.html
      
function imprime_laudos(id, tipo){
    $.get(`api-laudo-visu/${id}`, (res) => {
        let dados = res.data
        if (tipo == "Manutenção"){
            let unidade = dados.unidade
            let unidade_split = unidade.split('-')[0]
            let campus = unidade_split.split('-')[0].split(' ')[0]
      
            if (campus == "CAMPUS") {
              unidade_split = `${unidade_split.split(' ')[1]} ${unidade_split.split(' ')[2]}`
            } else {
              unidade_split = unidade_split.split(". ")[2]
            }
      
            let instituto = dados.instituto
            let instituto_split = instituto.split("-")[0]
            
            let criacao = dados.criacao
            let data = new Date(criacao)
      
            let dia = data.getDate();
            let mes = data.getMonth() + 1; 
            let ano = data.getFullYear();
            let hora = data.getHours();
            let minutos = data.getMinutes();
      
            if (dia < 10){
              dia = "0" + dia
            }
      
            if (minutos < 10){
              minutos = "0" + minutos
            }
      
            if (mes < 10){
              mes = "0" + mes
            }
      
            if (hora < 10){
              hora = "0" + hora
            }
                      
            let dataformatada = `${dia}/${mes}/${ano} - ${hora}:${minutos}`
            

            let content = ` 
                <!DOCTYPE html>
                    <html lang="pt-br">
                    <head>
                        <meta charset="UTF-8">
                        <meta name="viewport" content="width=device-width, initial-scale=1.0">
                        <link rel="stylesheet" type="text/css" href="style.css" media="print">
                        <link rel="preconnect" href="https://fonts.googleapis.com">
                        <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
                        <link href="https://fonts.googleapis.com/css2?family=Lato:wght@100&family=Roboto:wght@400&display=swap" rel="stylesheet">
                
                
                        <title>Laudo De Manutenção</title>
        
                        <style>
                    
                            body {
                            font-family: 'Roboto', sans-serif;
                            }
                    
                            .element {
                                
                                margin: 0;
                                padding: 0;
                            }
                    
                            .fixarRodape {
                            bottom: 0;
                            position: fixed;
                            width: 90%;
                            text-align: center;
                        }
                        </style>
                    </head>
      
                        <body>
                            <header>
                                <div style="display: flex; justify-content: space-between; align-items: center; border: 2px solid black; margin-bottom: 50px">
                                    <img style="padding: 15px;" height="70" width="70" src="static/imgs/puc_2.png" alt="imagem">
                                    <h3>SOLICITAÇÃO PARA MANUTENÇÃO</h3>
                                    <P style="padding-right: 10px;"> ${dataformatada}</P>
                                </div>
                            </header>
                    
                    
                            <div style=" margin-bottom: 40px;">
                                <div class="title" style="text-align: center; border-bottom: 1px solid black; margin-bottom: 20px;" >
                                    <h4 style="height: 1px; line-height: 3px;">DADOS DO SOLICITANTE</h4>
                                </div> 
                                <div style="display: flex; ">
                                    <div style="display: inline-block; width: 50%; padding-left: 5px; height: 60px;">
                                        <p style="font-size: 12px;"><b>Solicitante:</b></p>
                                        <p class="element" style="line-height: 1px;">${dados.nome_solicitante}</p>
                                    </div>
                                    <div style="display: block;  width: 50%; padding-left: 5px; height: 60px;">
                                        <p style="font-size: 12px;"><b>Email: </b></p>
                                        <p class="element" style="line-height: 2px;">${dados.email_solicitante}</p>
                                    </div>
                                </div>
                                <div style="display: flex; ">
                                    <div style="display: block; width: 40%;  padding-left:5px; height: 60px;">
                                        <p style="margin-right: 3px; font-size: 12px;"><b>Unidade:</b></p>
                                        <p class="element" style="line-height: 2px;">${unidade_split}</p>
                                    </div>
                                    <div style="display: block; width: 20%; padding-left:20px; height: 60px;">
                                        <p style="margin-right: 3px; font-size: 12px;"><b>Instituto:</b></p>
                                        <p class="element" style="line-height: 2px;">${instituto_split}</p>
                                    </div>
                                    <div style="display: block;  padding-left:5px; width:60%;">
                                    <div>
                                        <p style="margin-right: 3px; font-size: 15px"><b>Setor: </b><br>
                                        ${dados.setor}
                                        </p>
                                    </div>
                                    </div>
                                </div>
                                <div style="display: flex; ">
                                    <div style="display: block;  width: 25%; padding-left:5px; height: 60px;">
                                        <p style="margin-right: 3px; font-size: 12px;"><b>Nº Chamado:</b></p>
                                        <p class="element" style="line-height: 2px;">${dados.num_chamado}</p>
                                    </div>
                                    <div style="display: block; width: 25%; padding-left:5px; height: 60px;">
                                        <p style="margin-right: 3px; font-size: 12px;"><b>Prédio:</b></p>
                                        <p class="element" style="line-height: 2px;">${dados.num_predio}</p>
                                    </div>
                                    <div style="display: block; width: 25%; padding-left:5px; height: 60px;">
                                        <p style="margin-right: 3px; font-size: 12px;"><b>Sala:</b></p>
                                        <p class="element" style="line-height: 2px;">${dados.num_sala}</p>
                                    </div>
                                    <div style="display: block;  width: 25%; padding-left:5px; height: 60px;">
                                        <p style="margin-right: 3px; font-size: 12px;"><b>Ramal:</b></p>
                                        <p class="element" style="line-height: 2px;">${dados.num_ramal}</p>
                                    </div>
                                </div>
                            </div>    
      
                            <div>
                                <div style="text-align: center; border-bottom: 1px solid black; margin-bottom: 10px;">
                                    <h4 style="height: 1px; line-height: 3px;">DADOS DO EQUIPAMENTO</h4>
                                </div>
                                <div style="display: flex; ">
                                    <div style="display: block; width: 33%; padding-left:5px; height: 60px;">
                                        <p style="margin-right: 3px; font-size: 12px;"><b>Tipo: </b></p>
                                        <p class="element" style="line-height: 2px;">${dados.tipo_equip}</p>
                                    </div>
                                    <div style="display: block;  width: 50%; padding-left:5px; height: 60px;">
                                        <p style="margin-right: 3px; font-size: 12px;"><b>N° Patrimônio: </b></p>
                                        <p class="element" style="line-height: 2px;">${dados.patrimonio}</p>
                                    </div>
                                    <div style="display: block; width: 50%;  padding-left:5px; height: 60px;">
                                        <p style="margin-right: 3px; font-size: 12px;"><b>N° Série: </b></p>
                                        <p class="element" style="line-height: 2px;">${dados.num_serie}</p>
                                    </div>
                                </div>
      
                                <div style="display: flex; ">
                                    <div style="display: block; width: 40%; padding-left:5px; height: 60px;">
                                    <p style="margin-right: 3px; font-size: 12px;"><b>Marca: </b></p>
                                    <p class="element" style="line-height: 2px;">${dados.marca}</p>
                                    </div>
                                    <div style="display: block; width: 60%; padding-left:5px; height: 60px;">
                                        <p style="margin-right: 3px; font-size: 12px;"><b>Modelo: </b></p>
                                        <p class="element" style="line-height: 2px;">${dados.modelo}</p>
                                    </div>
                                </div>
                                <div style="display: flex; ">
                                    <div style="display: block;   width: 50%; padding-left:5px; height: 60px;">
                                        <p style="margin-right: 3px; font-size: 12px;"><b>Nome Centro de Custo: </b></p>
                                        <p class="element" style="line-height: 2px;">${dados.nome_centro_custo} </p>
                                    </div>
                                    <div style="display: block; width: 25%; padding-left:5px; height: 60px;">
                                        <p style="margin-right: 3px; font-size: 12px;"><b>Código Centro de Custo: </b></p>
                                        <p class="element" style="line-height: 2px;">${dados.centro_custo}</p>
                                    </div>
                                    <div style="display: block; width: 25%; padding-left:5px; height: 60px;">
                                        <p style="margin-right: 3px; font-size: 12px;"><b>Unidade Negócio:  </b></p>
                                        <p class="element" style="line-height: 2px;">${dados.unidade_negocio}</p>
                                    </div>
                                </div>
                            </div>
                            <div style="text-align: center; border-bottom: 1px solid black; margin-top: 60px">
                                <h4 style="height: 1px; line-height: 3px;">DIAGNÓSTICO</h4>
                            </div>
                            <div style="display: block;  width: 100%; padding-left:5px; height: 40px;">
                                <p style="margin-right: 3px; font-size: 12px;"><b>Técnico Responsável:</b></p>
                                <p class="element" style="line-height: 2px;">${dados.user}</p>
                            </div>
                            <div style="  padding-left:5px; width:100%; ">
                                <div style="padding-top: 0; margin-top: 0;">
                                    <p style="margin-right: 3px; font-size: 15px;"><b>Observações e testes realizados: </b><br>
                                    ${dados.observacoes}</p>
                                </div>
                            </div>
      
          
                            <footer class="fixarRodape">
                            <div style="display: flex; justify-content: space-around;">
                                <div style="width: 280px; text-align: center;">
                                    <hr style="border: 0; border-top: 1px solid black; background-color: black;">
                                    <b>Requisitante</b>
                                </div>
                                <div style="width: 280px; margin-left: 20px; text-align: center;">
                                    <hr style="border: 0; border-top: 1px solid black; background-color: black;">
                                    <b>Técnico </b>
                                </div>
                                <div style="width: 280px; margin-left: 20px; text-align: center;">
                                    <hr style="border: 0; border-top: 1px solid black; background-color: black;">
                                    <b>Divisão de Patrimônio</b>
                                </div>
                            </div>
                            </footer>
                        </body>
                     </html>
          `
        let win = window.open("", "_blank", "width=600, height=600")
      
        win.document.write(content)
        setTimeout(() => {
            win.print()
            win.close()
        }, 100)
        } else {
            let unidade = dados.unidade
            let unidade_split = unidade.split('-')[0]
            let campus = unidade_split.split('-')[0].split(' ')[0]
    
            if (campus == "CAMPUS") {
              unidade_split = `${unidade_split.split(' ')[1]} ${unidade_split.split(' ')[2]}`
            } else {
              unidade_split = unidade_split.split(". ")[2]
            }
            
            let instituto = dados.instituto
            let instituto_split = instituto.split("-")[0]
    
            let criacao = dados.criacao
            let data = new Date(criacao)
    
            var dia = data.getDate();
            var mes = data.getMonth() + 1; 
            var ano = data.getFullYear();
            var hora = data.getHours();
            var minutos = data.getMinutes();
            
            if (dia < 10){
              dia = "0" + dia
            }
      
            if (minutos < 10){
              minutos = "0" + minutos
            }
      
            if (mes < 10){
              mes = "0" + mes
            }
      
            if (hora < 10){
              hora = "0" + hora
            }
                    
            let dataformatada = `${dia}/${mes}/${ano} - ${hora}:${minutos}`
    
            let content = ` 
            <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <link rel="stylesheet" type="text/css" href="style.css" media="print">
            <link rel="preconnect" href="https://fonts.googleapis.com">
            <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
            <link href="https://fonts.googleapis.com/css2?family=Lato:wght@100&family=Roboto:wght@400&display=swap" rel="stylesheet">
    
    
            <title>Laudo De Devolução</title>
      
        <style>
    
          body {
            font-family: 'Roboto', sans-serif;
          }
    
          .element {
              
              margin: 0;
              padding: 0;
          }
    
          .fixarRodape {
            bottom: 0;
            position: fixed;
            width: 90%;
            text-align: center;
          }
        </style>
      </head>
    
        <body>
            <header>
                <div style="display: flex; justify-content: space-between; align-items: center; border: 2px solid black; margin-bottom: 50px">
                    <img style="padding: 15px;" height="70" width="70" src="static/imgs/puc_2.png" alt="imagem">
                    <h3>LAUDO DE DEVOLUÇÂO</h3>
                    <P style="padding-right: 10px;">${dataformatada}</P>
                </div>
            </header>
    
    
            <div style=" margin-bottom: 40px;">
              <div class="title" style="text-align: center; border-bottom: 1px solid black; margin-bottom: 20px;" >
                  <h4 style="height: 1px; line-height: 3px;">DADOS DO SOLICITANTE</h4>
              </div> 
              <div style="display: flex; ">
                  <div style="display: inline-block; width: 50%; padding-left: 5px; height: 60px;">
                      <p style="font-size: 12px;"><b>Solicitante:</b></p>
                      <p class="element" style="line-height: 1px;">${dados.nome_solicitante}</p>
                  </div>
                  <div style="display: block;  width: 50%; padding-left: 5px; height: 60px;">
                      <p style="font-size: 12px;"><b>Email: </b></p>
                      <p class="element" style="line-height: 2px;">${dados.email_solicitante}</p>
                  </div>
              </div>
              <div style="display: flex; ">
                  <div style="display: block; width: 40%;  padding-left:5px; height: 60px;">
                      <p style="margin-right: 3px; font-size: 12px;"><b>Unidade:</b></p>
                      <p class="element" style="line-height: 2px;">${unidade_split}</p>
                  </div>
                  <div style="display: block; width: 20%; padding-left:20px; height: 60px;">
                      <p style="margin-right: 3px; font-size: 12px;"><b>Instituto:</b></p>
                      <p class="element" style="line-height: 2px;">${instituto_split}</p>
                  </div>
                  <div style="display: block;  padding-left:5px; width:60%;">
                    <div>
                        <p style="margin-right: 3px; font-size: 15px"><b>Setor: </b><br>
                        ${dados.setor}
                        </p>
                    </div>
                  </div>
              </div>
              <div style="display: flex; ">
                  <div style="display: block;  width: 25%; padding-left:5px; height: 60px;">
                      <p style="margin-right: 3px; font-size: 12px;"><b>Nº Chamado:</b></p>
                      <p class="element" style="line-height: 2px;">${dados.num_chamado}</p>
                  </div>
                  <div style="display: block; width: 25%; padding-left:5px; height: 60px;">
                      <p style="margin-right: 3px; font-size: 12px;"><b>Prédio:</b></p>
                      <p class="element" style="line-height: 2px;">${dados.num_predio}</p>
                  </div>
                  <div style="display: block; width: 25%; padding-left:5px; height: 60px;">
                      <p style="margin-right: 3px; font-size: 12px;"><b>Sala:</b></p>
                      <p class="element" style="line-height: 2px;">${dados.num_sala}</p>
                  </div>
                  <div style="display: block;  width: 25%; padding-left:5px; height: 60px;">
                      <p style="margin-right: 3px; font-size: 12px;"><b>Ramal:</b></p>
                      <p class="element" style="line-height: 2px;">${dados.num_ramal}</p>
                  </div>
              </div>
          </div>    
        <div>
            <div style="text-align: center; border-bottom: 1px solid black; margin-bottom: 10px;">
                <h4 style="height: 1px; line-height: 3px;">DADOS DO EQUIPAMENTO</h4>
            </div>
            <div style="display: flex; ">
                <div style="display: block; width: 33%; padding-left:5px; height: 60px;">
                    <p style="margin-right: 3px; font-size: 12px;"><b>Tipo: </b></p>
                    <p class="element" style="line-height: 2px;">${dados.tipo_equip}</p>
                </div>
                <div style="display: block;  width: 33%; padding-left:5px; height: 60px;">
                  <p style="margin-right: 3px; font-size: 12px;"><b>N° Patrimônio: </b></p>
                  <p class="element" style="line-height: 2px;">${dados.patrimonio}</p>
              </div>
              <div style="display: block; width: 33%;  padding-left:5px; height: 60px;">
                  <p style="margin-right: 3px; font-size: 12px;"><b>N° Série: </b></p>
                  <p class="element" style="line-height: 2px;">${dados.num_serie}</p>
              </div>
            </div>
    
            <div style="display: flex; ">
                <div style="display: block; width: 40%; padding-left:5px; height: 60px;">
                  <p style="margin-right: 3px; font-size: 12px;"><b>Marca: </b></p>
                  <p class="element" style="line-height: 2px;">${dados.marca}</p>
              </div>
              <div style="display: block; width: 60%; padding-left:5px; height: 60px;">
                  <p style="margin-right: 3px; font-size: 12px;"><b>Modelo: </b></p>
                  <p class="element" style="line-height: 2px;">${dados.modelo}</p>
              </div>
            </div>
          </div>
    
    
          <div style="display: flex; ">
            <div style="display: block; width: 33%;  padding-left:5px; height: 60px;">
              <p style="margin-right: 3px; font-size: 12px;"><b>Tipo de Baixa </b></p>
              <p class="element" style="line-height: 2px;">${dados.tipo_baixa}</p>
            </div>
          </div>
        <div style="text-align: center; border-bottom: 1px solid black; margin-top: 40px">
          <h4 style="height: 1px; line-height: 3px;">DIAGNÓSTICO</h4>
        </div>
        <div style="display: block;  width: 100%; padding-left:5px; height: 40px;">
            <p style="margin-right: 3px; font-size: 12px;"><b>Técnico Responsável:</b></p>
            <p class="element" style="line-height: 2px;">${dados.user}</p>
        </div>
        <div style="  padding-left:5px; width:100%; ">
          <div style="padding-top: 0; margin-top: 0;">
              <p style="margin-right: 3px; font-size: 15px;"><b>Observações e testes realizados: </b><br>
              ${dados.observacoes}</p>
          </div>
        </div>
        
        <footer class="fixarRodape">
          <div style="display: flex; justify-content: space-around; margin-top: 50px;">
              <div style="width: 280px; text-align: center;">
                  <hr style="border: 0; border-top: 1px solid black; background-color: black;">
                  <b>Requisitante</b>
              </div>
              <div style="width: 280px; margin-left: 20px; text-align: center;">
                  <hr style="border: 0; border-top: 1px solid black; background-color: black;">
                  <b>Técnico </b>
              </div>
              <div style="width: 280px; margin-left: 20px; text-align: center;">
                  <hr style="border: 0; border-top: 1px solid black; background-color: black;">
                  <b>Divisão de Patrimônio</b>
              </div>
          </div>
        </footer>
        </body>
    
    
    
        </html>
        `
            let win = window.open("", "_blank", "width=600, height=600") 
            
            win.document.write(content)
            setTimeout(() => {
              win.print()
              win.close()
            }, 100)
        }
    })
}






        $(document).ready(function(){
            $('#btn-imprime-checks').on('click', function() {
        
                let checkbox = $('input:checkbox:checked').not('.check-todos')
                let id
                let tipoLaudo 
                let aux_array = []
                let laudos = []

                checkbox.each(function() {
                    id = $(this).attr('id')
                    tipoLaudo = $(this).closest('tr').find('td:eq(4)').text();


                    

                    if(tipoLaudo == 'Manutenção'){
        
                        var aux_content = $.get(`api-laudo-visu/${id}`).then((res) => {
                        let dados = res.data
                        let unidade = dados.unidade
                        let unidade_split = unidade.split('-')[0]
                        let campus = unidade_split.split('-')[0].split(' ')[0]
                
                        if (campus == "CAMPUS") {
                        unidade_split = `${unidade_split.split(' ')[1]} ${unidade_split.split(' ')[2]}`
                        } else {
                        unidade_split = unidade_split.split(". ")[2]
                        }
                
                        let instituto = dados.instituto
                        let instituto_split = instituto.split("-")[0]
                        
                        let criacao = dados.criacao
                        let data = new Date(criacao)
                
                        let dia = data.getDate();
                        let mes = data.getMonth() + 1; 
                        let ano = data.getFullYear();
                        let hora = data.getHours();
                        let minutos = data.getMinutes();
                
                        if (dia < 10){
                        dia = "0" + dia
                        }
                
                        if (minutos < 10){
                        minutos = "0" + minutos
                        }
                
                        if (mes < 10){
                        mes = "0" + mes
                        }
                
                        if (hora < 10){
                        hora = "0" + hora
                        }
                                
                        let dataformatada = `${dia}/${mes}/${ano} - ${hora}:${minutos}`
        
                                var content = `
                                <!DOCTYPE html>
                                    <html lang="pt-br">
                                    <head>
                                        <meta charset="UTF-8">
                                        <meta name="viewport" content="width=device-width, initial-scale=1.0">
                                        <link rel="stylesheet" type="text/css" href="style.css" media="print">
                                        <link rel="preconnect" href="https://fonts.googleapis.com">
                                        <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
                                        <link href="https://fonts.googleapis.com/css2?family=Lato:wght@100&family=Roboto:wght@400&display=swap" rel="stylesheet">
                                
                                
                                        <title>Laudo De Manutenção</title>
                        
                                        <style>
                                    
                                            body {
                                            font-family: 'Roboto', sans-serif;
                                            }
                                    
                                            .element {
                                                
                                                margin: 0;
                                                padding: 0;
                                            }
                                    
                                            .fixarRodape {
                                            bottom: 0;
                                            position: fixed;
                                            width: 90%;
                                            text-align: center;
                                        }
                                        </style>
                                    </head>
                    
                                        <body>
                                            <header>
                                                <div style="display: flex; justify-content: space-between; align-items: center; border: 2px solid black; margin-bottom: 50px">
                                                    <img style="padding: 15px;" height="70" width="70" src="static/imgs/puc_2.png" alt="imagem">
                                                    <h3>SOLICITAÇÃO PARA MANUTENÇÃO</h3>
                                                    <P style="padding-right: 10px;"> ${dataformatada}</P>
                                                </div>
                                            </header>
                                    
                                    
                                            <div style=" margin-bottom: 40px;">
                                                <div class="title" style="text-align: center; border-bottom: 1px solid black; margin-bottom: 20px;" >
                                                    <h4 style="height: 1px; line-height: 3px;">DADOS DO SOLICITANTE</h4>
                                                </div> 
                                                <div style="display: flex; ">
                                                    <div style="display: inline-block; width: 50%; padding-left: 5px; height: 60px;">
                                                        <p style="font-size: 12px;"><b>Solicitante:</b></p>
                                                        <p class="element" style="line-height: 1px;">${dados.nome_solicitante}</p>
                                                    </div>
                                                    <div style="display: block;  width: 50%; padding-left: 5px; height: 60px;">
                                                        <p style="font-size: 12px;"><b>Email: </b></p>
                                                        <p class="element" style="line-height: 2px;">${dados.email_solicitante}</p>
                                                    </div>
                                                </div>
                                                <div style="display: flex; ">
                                                    <div style="display: block; width: 40%;  padding-left:5px; height: 60px;">
                                                        <p style="margin-right: 3px; font-size: 12px;"><b>Unidade:</b></p>
                                                        <p class="element" style="line-height: 2px;">${unidade_split}</p>
                                                    </div>
                                                    <div style="display: block; width: 20%; padding-left:20px; height: 60px;">
                                                        <p style="margin-right: 3px; font-size: 12px;"><b>Instituto:</b></p>
                                                        <p class="element" style="line-height: 2px;">${instituto_split}</p>
                                                    </div>
                                                    <div style="display: block;  padding-left:5px; width:60%;">
                                                    <div>
                                                        <p style="margin-right: 3px; font-size: 15px"><b>Setor: </b><br>
                                                        ${dados.setor}
                                                        </p>
                                                    </div>
                                                    </div>
                                                </div>
                                                <div style="display: flex; ">
                                                    <div style="display: block;  width: 25%; padding-left:5px; height: 60px;">
                                                        <p style="margin-right: 3px; font-size: 12px;"><b>Nº Chamado:</b></p>
                                                        <p class="element" style="line-height: 2px;">${dados.num_chamado}</p>
                                                    </div>
                                                    <div style="display: block; width: 25%; padding-left:5px; height: 60px;">
                                                        <p style="margin-right: 3px; font-size: 12px;"><b>Prédio:</b></p>
                                                        <p class="element" style="line-height: 2px;">${dados.num_predio}</p>
                                                    </div>
                                                    <div style="display: block; width: 25%; padding-left:5px; height: 60px;">
                                                        <p style="margin-right: 3px; font-size: 12px;"><b>Sala:</b></p>
                                                        <p class="element" style="line-height: 2px;">${dados.num_sala}</p>
                                                    </div>
                                                    <div style="display: block;  width: 25%; padding-left:5px; height: 60px;">
                                                        <p style="margin-right: 3px; font-size: 12px;"><b>Ramal:</b></p>
                                                        <p class="element" style="line-height: 2px;">${dados.num_ramal}</p>
                                                    </div>
                                                </div>
                                            </div>    
                    
                                            <div>
                                                <div style="text-align: center; border-bottom: 1px solid black; margin-bottom: 10px;">
                                                    <h4 style="height: 1px; line-height: 3px;">DADOS DO EQUIPAMENTO</h4>
                                                </div>
                                                <div style="display: flex; ">
                                                    <div style="display: block; width: 33%; padding-left:5px; height: 60px;">
                                                        <p style="margin-right: 3px; font-size: 12px;"><b>Tipo: </b></p>
                                                        <p class="element" style="line-height: 2px;">${dados.tipo_equip}</p>
                                                    </div>
                                                    <div style="display: block;  width: 50%; padding-left:5px; height: 60px;">
                                                        <p style="margin-right: 3px; font-size: 12px;"><b>N° Patrimônio: </b></p>
                                                        <p class="element" style="line-height: 2px;">${dados.patrimonio}</p>
                                                    </div>
                                                    <div style="display: block; width: 50%;  padding-left:5px; height: 60px;">
                                                        <p style="margin-right: 3px; font-size: 12px;"><b>N° Série: </b></p>
                                                        <p class="element" style="line-height: 2px;">${dados.num_serie}</p>
                                                    </div>
                                                </div>
                    
                                                <div style="display: flex; ">
                                                    <div style="display: block; width: 40%; padding-left:5px; height: 60px;">
                                                    <p style="margin-right: 3px; font-size: 12px;"><b>Marca: </b></p>
                                                    <p class="element" style="line-height: 2px;">${dados.marca}</p>
                                                    </div>
                                                    <div style="display: block; width: 60%; padding-left:5px; height: 60px;">
                                                        <p style="margin-right: 3px; font-size: 12px;"><b>Modelo: </b></p>
                                                        <p class="element" style="line-height: 2px;">${dados.modelo}</p>
                                                    </div>
                                                </div>
                                                <div style="display: flex; ">
                                                    <div style="display: block;   width: 50%; padding-left:5px; height: 60px;">
                                                        <p style="margin-right: 3px; font-size: 12px;"><b>Nome Centro de Custo: </b></p>
                                                        <p class="element" style="line-height: 2px;">${dados.nome_centro_custo} </p>
                                                    </div>
                                                    <div style="display: block; width: 25%; padding-left:5px; height: 60px;">
                                                        <p style="margin-right: 3px; font-size: 12px;"><b>Código Centro de Custo: </b></p>
                                                        <p class="element" style="line-height: 2px;">${dados.centro_custo}</p>
                                                    </div>
                                                    <div style="display: block; width: 25%; padding-left:5px; height: 60px;">
                                                        <p style="margin-right: 3px; font-size: 12px;"><b>Unidade Negócio:  </b></p>
                                                        <p class="element" style="line-height: 2px;">${dados.unidade_negocio}</p>
                                                    </div>
                                                </div>
                                            </div>
                                            <div style="text-align: center; border-bottom: 1px solid black; margin-top: 60px">
                                                <h4 style="height: 1px; line-height: 3px;">DIAGNÓSTICO</h4>
                                            </div>
                                            <div style="display: block;  width: 100%; padding-left:5px; height: 40px;">
                                                <p style="margin-right: 3px; font-size: 12px;"><b>Técnico Responsável:</b></p>
                                                <p class="element" style="line-height: 2px;">${dados.user}</p>
                                            </div>
                                            <div style="  padding-left:5px; width:100%; ">
                                                <div style="padding-top: 0; margin-top: 0;">
                                                    <p style="margin-right: 3px; font-size: 15px;"><b>Observações e testes realizados: </b><br>
                                                    ${dados.observacoes}</p>
                                                </div>
                                            </div>
                    
                        
                                            
                                            <div style="display: flex; justify-content: space-around; margin-top: 100px;">
                                                <div style="width: 280px; text-align: center;">
                                                    <hr style="border: 0; border-top: 1px solid black; background-color: black;">
                                                    <b>Requisitante</b>
                                                </div>
                                                <div style="width: 280px; margin-left: 20px; text-align: center;">
                                                    <hr style="border: 0; border-top: 1px solid black; background-color: black;">
                                                    <b>Técnico </b>
                                                </div>
                                                <div style="width: 280px; margin-left: 20px; text-align: center;">
                                                    <hr style="border: 0; border-top: 1px solid black; background-color: black;">
                                                    <b>Divisão de Patrimônio</b>
                                                </div>
                                            </div>
                                            
                                        </body>
                                    </html>
                                `
                                laudos.push(content)
                            })
                            aux_array.push(aux_content)
                       
                    }  else {
                        var aux_content = $.get(`api-laudo-visu/${id}`).then((res) => {
                            let dados = res.data
                            let unidade = dados.unidade
                            let unidade_split = unidade.split('-')[0]
                            let campus = unidade_split.split('-')[0].split(' ')[0]
                    
                            if (campus == "CAMPUS") {
                            unidade_split = `${unidade_split.split(' ')[1]} ${unidade_split.split(' ')[2]}`
                            } else {
                            unidade_split = unidade_split.split(". ")[2]
                            }
                            
                            let instituto = dados.instituto
                            let instituto_split = instituto.split("-")[0]
                    
                            let criacao = dados.criacao
                            let data = new Date(criacao)
                    
                            var dia = data.getDate();
                            var mes = data.getMonth() + 1; 
                            var ano = data.getFullYear();
                            var hora = data.getHours();
                            var minutos = data.getMinutes();
                            
                            if (dia < 10){
                            dia = "0" + dia
                            }
                    
                            if (minutos < 10){
                            minutos = "0" + minutos
                            }
                    
                            if (mes < 10){
                            mes = "0" + mes
                            }
                    
                            if (hora < 10){
                            hora = "0" + hora
                            }
                                    
                            let dataformatada = `${dia}/${mes}/${ano} - ${hora}:${minutos}`
                    
                            let content = ` 
                            <!DOCTYPE html>
                        <html lang="en">
                        <head>
                            <meta charset="UTF-8">
                            <meta name="viewport" content="width=device-width, initial-scale=1.0">
                            <link rel="stylesheet" type="text/css" href="style.css" media="print">
                            <link rel="preconnect" href="https://fonts.googleapis.com">
                            <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
                            <link href="https://fonts.googleapis.com/css2?family=Lato:wght@100&family=Roboto:wght@400&display=swap" rel="stylesheet">
                    
                    
                            <title>Laudo De Devolução</title>
                    
                        <style>
                    
                        body {
                            font-family: 'Roboto', sans-serif;
                        }
                    
                        .element {
                            
                            margin: 0;
                            padding: 0;
                        }
                    
                        .fixarRodape {
                            bottom: 0;
                            position: fixed;
                            width: 90%;
                            text-align: center;
                        }
                        </style>
                    </head>
                    
                        <body>
                            <header>
                                <div style="display: flex; justify-content: space-between; align-items: center; border: 2px solid black; margin-bottom: 50px">
                                    <img style="padding: 15px;" height="70" width="70" src="static/imgs/puc_2.png" alt="imagem">
                                    <h3>LAUDO DE DEVOLUÇÂO</h3>
                                    <P style="padding-right: 10px;">${dataformatada}</P>
                                </div>
                            </header>
                    
                    
                            <div style=" margin-bottom: 40px;">
                            <div class="title" style="text-align: center; border-bottom: 1px solid black; margin-bottom: 20px;" >
                                <h4 style="height: 1px; line-height: 3px;">DADOS DO SOLICITANTE</h4>
                            </div> 
                            <div style="display: flex; ">
                                <div style="display: inline-block; width: 50%; padding-left: 5px; height: 60px;">
                                    <p style="font-size: 12px;"><b>Solicitante:</b></p>
                                    <p class="element" style="line-height: 1px;">${dados.nome_solicitante}</p>
                                </div>
                                <div style="display: block;  width: 50%; padding-left: 5px; height: 60px;">
                                    <p style="font-size: 12px;"><b>Email: </b></p>
                                    <p class="element" style="line-height: 2px;">${dados.email_solicitante}</p>
                                </div>
                            </div>
                            <div style="display: flex; ">
                                <div style="display: block; width: 40%;  padding-left:5px; height: 60px;">
                                    <p style="margin-right: 3px; font-size: 12px;"><b>Unidade:</b></p>
                                    <p class="element" style="line-height: 2px;">${unidade_split}</p>
                                </div>
                                <div style="display: block; width: 20%; padding-left:20px; height: 60px;">
                                    <p style="margin-right: 3px; font-size: 12px;"><b>Instituto:</b></p>
                                    <p class="element" style="line-height: 2px;">${instituto_split}</p>
                                </div>
                                <div style="display: block;  padding-left:5px; width:60%;">
                                    <div>
                                        <p style="margin-right: 3px; font-size: 15px"><b>Setor: </b><br>
                                        ${dados.setor}
                                        </p>
                                    </div>
                                </div>
                            </div>
                            <div style="display: flex; ">
                                <div style="display: block;  width: 25%; padding-left:5px; height: 60px;">
                                    <p style="margin-right: 3px; font-size: 12px;"><b>Nº Chamado:</b></p>
                                    <p class="element" style="line-height: 2px;">${dados.num_chamado}</p>
                                </div>
                                <div style="display: block; width: 25%; padding-left:5px; height: 60px;">
                                    <p style="margin-right: 3px; font-size: 12px;"><b>Prédio:</b></p>
                                    <p class="element" style="line-height: 2px;">${dados.num_predio}</p>
                                </div>
                                <div style="display: block; width: 25%; padding-left:5px; height: 60px;">
                                    <p style="margin-right: 3px; font-size: 12px;"><b>Sala:</b></p>
                                    <p class="element" style="line-height: 2px;">${dados.num_sala}</p>
                                </div>
                                <div style="display: block;  width: 25%; padding-left:5px; height: 60px;">
                                    <p style="margin-right: 3px; font-size: 12px;"><b>Ramal:</b></p>
                                    <p class="element" style="line-height: 2px;">${dados.num_ramal}</p>
                                </div>
                            </div>
                        </div>    
                        
                    
                    
                    
                        <div>
                            <div style="text-align: center; border-bottom: 1px solid black; margin-bottom: 10px;">
                                <h4 style="height: 1px; line-height: 3px;">DADOS DO EQUIPAMENTO</h4>
                            </div>
                            <div style="display: flex; ">
                                <div style="display: block; width: 33%; padding-left:5px; height: 60px;">
                                    <p style="margin-right: 3px; font-size: 12px;"><b>Tipo: </b></p>
                                    <p class="element" style="line-height: 2px;">${dados.tipo_equip}</p>
                                </div>
                                <div style="display: block;  width: 33%; padding-left:5px; height: 60px;">
                                <p style="margin-right: 3px; font-size: 12px;"><b>N° Patrimônio: </b></p>
                                <p class="element" style="line-height: 2px;">${dados.patrimonio}</p>
                            </div>
                            <div style="display: block; width: 33%;  padding-left:5px; height: 60px;">
                                <p style="margin-right: 3px; font-size: 12px;"><b>N° Série: </b></p>
                                <p class="element" style="line-height: 2px;">${dados.num_serie}</p>
                            </div>
                            </div>
                    
                            <div style="display: flex; ">
                                <div style="display: block; width: 40%; padding-left:5px; height: 60px;">
                                <p style="margin-right: 3px; font-size: 12px;"><b>Marca: </b></p>
                                <p class="element" style="line-height: 2px;">${dados.marca}</p>
                            </div>
                            <div style="display: block; width: 60%; padding-left:5px; height: 60px;">
                                <p style="margin-right: 3px; font-size: 12px;"><b>Modelo: </b></p>
                                <p class="element" style="line-height: 2px;">${dados.modelo}</p>
                            </div>
                            </div>
                        </div>
                    
                    
                        <div style="display: flex; ">
                            <div style="display: block; width: 33%;  padding-left:5px; height: 60px;">
                            <p style="margin-right: 3px; font-size: 12px;"><b>Tipo de Baixa </b></p>
                            <p class="element" style="line-height: 2px;">${dados.tipo_baixa}</p>
                            </div>
                        </div>
                        <div style="text-align: center; border-bottom: 1px solid black; margin-top: 40px">
                        <h4 style="height: 1px; line-height: 3px;">DIAGNÓSTICO</h4>
                        </div>
                        <div style="display: block;  width: 100%; padding-left:5px; height: 40px;">
                            <p style="margin-right: 3px; font-size: 12px;"><b>Técnico Responsável:</b></p>
                            <p class="element" style="line-height: 2px;">${dados.user}</p>
                        </div>
                        <div style="  padding-left:5px; width:100%; ">
                        <div style="padding-top: 0; margin-top: 0;">
                            <p style="margin-right: 3px; font-size: 15px;"><b>Observações e testes realizados: </b><br>
                            ${dados.observacoes}</p>
                        </div>
                        </div>
                        
                        
                        <div style="display: flex; justify-content: space-around; margin-top: 100px;">
                            <div style="width: 280px; text-align: center;">
                                <hr style="border: 0; border-top: 1px solid black; background-color: black;">
                                <b>Requisitante</b>
                            </div>
                            <div style="width: 280px; margin-left: 20px; text-align: center;">
                                <hr style="border: 0; border-top: 1px solid black; background-color: black;">
                                <b>Técnico </b>
                            </div>
                            <div style="width: 280px; margin-left: 20px; text-align: center;">
                                <hr style="border: 0; border-top: 1px solid black; background-color: black;">
                                <b>Divisão de Patrimônio</b>
                            </div>
                        </div>
                        
                        </body>
                    
                    
                    
                        </html>
                        `
                        laudos.push(content)
                        })
                        aux_array.push(aux_content)
                    }
                })

        
                
                $.when.apply($, aux_array).then(function() {
                    imprimirLaudo(laudos)
                })
        
            })
        })
        
        
        function imprimirLaudo(laudos) {
            
                let conteudo = laudos.join("<div style='page-break-before: always;'></div>")
                let janela = window.open('', '', 'width=800,height=800')
                janela.document.write(conteudo)
                janela.document.close()
                setTimeout(() => {
                    janela.print()
                    janela.close()
                }, 400)
                
            
        }
    