<script setup>

const props = defineProps(['next', 'previous', 'count', 'current', 'total_pages'])
const emit = defineEmits(['nextPage', 'previousPage', 'changePage']);

function rate_next_pages(page, current_page, total_pages) {
   if (page === current_page) {
      return true
   }

   else if (current_page === page - 1 && page != 3) {
      return true
   }

   else if (current_page === page + 1) {
      return true
   }

   else if (page <= total_pages && page > total_pages - 2) {
      return true
   }

   else {
      return false
   }
}
function rate_first_last_pages(page, current_page, total_pages,) {
   if (page == total_pages - 2 && current_page > 2 && (current_page != total_pages)) {
      return true
   }
   else { return false }
}
function rate_last_pages(page, current_page, total_pages) {
   if (current_page != total_pages - 1 && current_page != total_pages - 2) {
      if (page <= total_pages && page > total_pages - 2) { return true }
      else { return false }
   }
   else { return false }
}



</script>
<template>

   <div class="d-flex" v-if="props.total_pages > 0">



      <button class="btn btn-estilo" @click.prevent="$emit('previousPage')"
         :disabled="props.current === 0">Anterior</button>

      <div class="d-flex gap-2 align-items-center" v-for="n in props.total_pages" :key="n">

         <button v-if="props.total_pages < 7" :class="(props.current + 1) == n ? 'page-select' : ''" class=" page-estilo"
            @click.prevent="$emit('changePage', (n - 1))" :disabled="(props.current + 1) == n">
            {{ n }}
         </button>


         <div v-else class="d-flex align-items-center ">
            <div v-if="n < 3" class="d-flex align-items-center ">

               <button :class="(props.current + 1) == n ? 'page-select' : ''" class=" page-estilo"
                  @click.prevent="$emit('changePage', (n - 1))" :disabled="(props.current + 1) == n">
                  {{ n }}
               </button>

               <p v-if="n === 2" class="m-0 p-0">...</p>
            </div>

            <div v-else >

               <button v-if="rate_next_pages(n, props.current + 1, props.total_pages)"
                  :class="(props.current + 1) == n ? 'page-select' : ''" class=" page-estilo"
                  @click.prevent="$emit('changePage', (n - 1))" :disabled="(props.current + 1) == n">
                  {{ n }}
               </button>

               <div v-else>

                  <p v-if="rate_first_last_pages(n, props.current + 1, props.total_pages)" class="m-0 p-0">...</p>


                  <button v-if="rate_last_pages(n, props.current + 1, props.total_pages)"
                     :class="(props.current + 1) == n ? 'page-select' : ''" class=" page-estilo"
                     @click.prevent="$emit('changePage', (n - 1))" :disabled="(props.current + 1) == n">
                     {{ n }}
                  </button>
               </div>
            </div>


         </div>

      </div>


      <button @click.prevent="$emit('nextPage')" class="btn btn-estilo" :disabled="!props.next">Próxima</button>
   </div>
   


</template>
<style scoped>
.page-select {
  background-color: #206BC4 !important;
  color: white !important;
  border-radius: 7px;
}
.page-estilo {
  border: none;
  background: 0 0;
  padding: 1px 10px;
}

.page-estilo:hover {
  color: rgb(86, 148, 242);
}
.btn-estilo {
  background: 0 0;
  font-size: 10px;
  max-width: 50px;
  border: none;
  transition: color 200ms ease-in-out;

}

.btn-estilo:hover {

  color: rgb(86, 148, 242);
  border: var(--tblr-pagination-border-width) solid var(--tblr-pagination-border-color);
}
</style>