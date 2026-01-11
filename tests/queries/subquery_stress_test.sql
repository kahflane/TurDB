-- Subquery Stress Test Suite
-- Comprehensive test suite to find bugs in the query system
-- Tests: subqueries, JOINs, aggregations, CTEs, HAVING, date filters, expressions
-- STRICT expectations - failures indicate bugs or missing features

-- =============================================================================
-- SECTION 1: Basic Subquery Tests
-- =============================================================================

-- Q1: Simple subquery with LIMIT
SELECT * FROM (SELECT id, name FROM organizations LIMIT 5);
-- EXPECT_ROWS: 5

-- Q2: Subquery with ORDER BY DESC + LIMIT (critical test)
SELECT * FROM (SELECT id, name FROM organizations ORDER BY id DESC LIMIT 3);
-- EXPECT_ROWS: 3

-- Q3: Subquery with ORDER BY ASC + LIMIT
SELECT * FROM (SELECT id, name FROM organizations ORDER BY id ASC LIMIT 3);
-- EXPECT_ROWS: 3

-- Q4: Verify ORDER BY DESC returns highest ID first
SELECT id FROM (SELECT id FROM organizations ORDER BY id DESC LIMIT 1);
-- EXPECT_MAX_ID

-- Q5: Verify ORDER BY ASC returns lowest ID first
SELECT id FROM (SELECT id FROM organizations ORDER BY id ASC LIMIT 1);
-- EXPECT_MIN_ID

-- =============================================================================
-- SECTION 2: Nested Subqueries (Multiple Levels)
-- =============================================================================

-- Q6: 2-level nesting - inner DESC, outer takes subset
SELECT * FROM (SELECT * FROM (SELECT id, name FROM organizations ORDER BY id DESC LIMIT 20) AS inner_sub LIMIT 5) AS outer_sub;
-- EXPECT_ROWS: 5

-- Q7: 2-level nesting with ORDER reversal
SELECT * FROM (SELECT * FROM (SELECT id, name FROM organizations ORDER BY id ASC LIMIT 50) AS inner_sub ORDER BY id DESC LIMIT 10) AS outer_sub;
-- EXPECT_ROWS: 10

-- Q8: 3-level deep nesting
SELECT * FROM (SELECT * FROM (SELECT * FROM (SELECT id, name FROM organizations LIMIT 100) AS l1 LIMIT 50) AS l2 LIMIT 10) AS l3;
-- EXPECT_ROWS: 10

-- Q9: 4-level deep nesting stress test
SELECT * FROM (SELECT * FROM (SELECT * FROM (SELECT * FROM (SELECT id, name FROM organizations ORDER BY id DESC LIMIT 200) AS l1 LIMIT 100) AS l2 LIMIT 50) AS l3 LIMIT 10) AS l4;
-- EXPECT_ROWS: 10

-- Q10: 5-level deep nesting
SELECT * FROM (SELECT * FROM (SELECT * FROM (SELECT * FROM (SELECT * FROM (SELECT id, name FROM organizations LIMIT 100) AS l1 LIMIT 80) AS l2 LIMIT 60) AS l3 LIMIT 40) AS l4 LIMIT 20) AS l5;
-- EXPECT_ROWS: 20

-- =============================================================================
-- SECTION 3: CTE (Common Table Expressions) - WITH Clause
-- =============================================================================

-- Q11: Simple CTE
WITH org_subset AS (SELECT id, name FROM organizations LIMIT 10)
SELECT * FROM org_subset;
-- EXPECT_ROWS: 10

-- Q12: CTE with filter
WITH filtered_orgs AS (SELECT id, name FROM organizations WHERE id > 100)
SELECT * FROM filtered_orgs LIMIT 10;
-- EXPECT_ROWS: 10

-- Q13: CTE with aggregation
WITH org_stats AS (SELECT COUNT(*) as total, MAX(id) as max_id FROM organizations)
SELECT * FROM org_stats;
-- EXPECT_ROWS: 1

-- Q14: Multiple CTEs
WITH
    large_orgs AS (SELECT id, name FROM organizations WHERE id > 1000),
    small_orgs AS (SELECT id, name FROM organizations WHERE id < 100)
SELECT * FROM large_orgs LIMIT 5;
-- EXPECT_ROWS: 5

-- Q15: CTE referencing another CTE
WITH
    base AS (SELECT id, name FROM organizations LIMIT 100),
    filtered AS (SELECT * FROM base WHERE id > 50)
SELECT * FROM filtered LIMIT 10;
-- EXPECT_ROWS: 10

-- Q16: CTE with GROUP BY
WITH grouped AS (
    SELECT creation_date, COUNT(*) as cnt
    FROM organizations
    GROUP BY creation_date
)
SELECT * FROM grouped ORDER BY cnt DESC LIMIT 5;
-- EXPECT_ROWS: 5

-- Q17: Recursive-style CTE pattern (non-recursive)
WITH RECURSIVE numbered AS (
    SELECT id, name, 1 as level FROM organizations WHERE id < 10
)
SELECT * FROM numbered LIMIT 10;
-- EXPECT_ROWS: >= 1

-- =============================================================================
-- SECTION 4: HAVING Clause
-- =============================================================================

-- Q18: Simple HAVING
SELECT creation_date, COUNT(*) as cnt FROM organizations GROUP BY creation_date HAVING COUNT(*) > 1 LIMIT 10;
-- EXPECT_ROWS: >= 1

-- Q19: HAVING with multiple conditions
SELECT creation_date, COUNT(*) as cnt, MIN(id) as min_id FROM organizations GROUP BY creation_date HAVING COUNT(*) > 1 AND MIN(id) > 10 LIMIT 10;
-- EXPECT_ROWS: >= 1

-- Q20: HAVING with aggregate comparison
SELECT creation_date, COUNT(*) as cnt FROM organizations GROUP BY creation_date HAVING COUNT(*) >= 2 LIMIT 10;
-- EXPECT_ROWS: >= 1

-- Q21: HAVING in subquery
SELECT * FROM (SELECT creation_date, COUNT(*) as cnt FROM organizations GROUP BY creation_date HAVING COUNT(*) > 1) AS sub LIMIT 10;
-- EXPECT_ROWS: >= 1

-- Q22: HAVING with SUM
SELECT reward_type, SUM(reward_quantity) as total FROM competitions GROUP BY reward_type HAVING SUM(reward_quantity) > 1000 LIMIT 5;
-- EXPECT_ROWS: >= 1

-- Q23: HAVING with AVG
SELECT reward_type, AVG(total_teams) as avg_teams FROM competitions GROUP BY reward_type HAVING AVG(total_teams) > 10 LIMIT 5;
-- EXPECT_ROWS: >= 1

-- Q24: Complex HAVING with OR
SELECT creation_date, COUNT(*) as cnt FROM organizations GROUP BY creation_date HAVING COUNT(*) > 5 OR COUNT(*) = 1 LIMIT 10;
-- EXPECT_ROWS: >= 1

-- Q25: HAVING with expression
SELECT creation_date, COUNT(*) as cnt FROM organizations GROUP BY creation_date HAVING COUNT(*) * 2 > 4 LIMIT 10;
-- EXPECT_ROWS: >= 1

-- =============================================================================
-- SECTION 5: Date Filters and Operations
-- =============================================================================

-- Q26: Date equality (specific date may not exist)
SELECT * FROM (SELECT id, name, creation_date FROM organizations WHERE creation_date = '01/01/2020' LIMIT 10);
-- EXPECT_ROWS: >= 0

-- Q27: Date comparison (greater than)
SELECT * FROM (SELECT id, name, creation_date FROM organizations WHERE creation_date > '01/01/2010' LIMIT 10);
-- EXPECT_ROWS: >= 1

-- Q28: Date comparison (less than)
SELECT * FROM (SELECT id, name, creation_date FROM organizations WHERE creation_date < '01/01/2025' LIMIT 10);
-- EXPECT_ROWS: >= 1

-- Q29: Date range with BETWEEN
SELECT * FROM (SELECT id, name, creation_date FROM organizations WHERE creation_date BETWEEN '01/01/2010' AND '12/31/2025' LIMIT 20);
-- EXPECT_ROWS: >= 1

-- Q30: Date in ORDER BY
SELECT * FROM (SELECT id, name, creation_date FROM organizations ORDER BY creation_date DESC LIMIT 10);
-- EXPECT_ROWS: 10

-- Q31: Date in GROUP BY
SELECT * FROM (SELECT creation_date, COUNT(*) as cnt FROM organizations GROUP BY creation_date ORDER BY creation_date DESC LIMIT 10);
-- EXPECT_ROWS: >= 1

-- Q32: Datetime comparison on competitions
SELECT * FROM (SELECT id, title, enabled_date FROM competitions WHERE enabled_date > '01/01/2010' ORDER BY enabled_date DESC LIMIT 10);
-- EXPECT_ROWS: >= 1

-- Q33: Multiple date conditions
SELECT * FROM (SELECT id, title, enabled_date, deadline_date FROM competitions WHERE enabled_date > '01/01/2010' AND deadline_date < '01/01/2030' LIMIT 10);
-- EXPECT_ROWS: >= 1

-- Q34: Date with NULL check
SELECT * FROM (SELECT id, title, deadline_date FROM competitions WHERE deadline_date IS NOT NULL ORDER BY deadline_date DESC LIMIT 10);
-- EXPECT_ROWS: >= 1

-- Q35: Aggregate on date-filtered data
SELECT * FROM (SELECT COUNT(*) as cnt, MIN(creation_date) as earliest, MAX(creation_date) as latest FROM organizations WHERE creation_date > '01/01/2000');
-- EXPECT_ROWS: 1

-- =============================================================================
-- SECTION 6: JOIN Operations
-- =============================================================================

-- Q36: Simple two-table JOIN
SELECT o.id, o.name, uo.user_id FROM organizations o, user_organizations uo WHERE o.id = uo.organization_id LIMIT 10;
-- EXPECT_ROWS: >= 1

-- Q37: JOIN with ORDER BY
SELECT o.id, o.name, c.title FROM organizations o, competitions c WHERE c.organization_id = o.id ORDER BY o.id DESC LIMIT 10;
-- EXPECT_ROWS: >= 1

-- Q38: JOIN with aggregate
SELECT o.id, o.name, COUNT(*) as member_count FROM organizations o, user_organizations uo WHERE o.id = uo.organization_id GROUP BY o.id, o.name LIMIT 10;
-- EXPECT_ROWS: >= 1

-- Q39: Self-join pattern via subqueries
SELECT a.id, b.id FROM (SELECT id FROM organizations LIMIT 5) AS a, (SELECT id FROM organizations LIMIT 5) AS b WHERE a.id < b.id;
-- EXPECT_ROWS: 10

-- Q40: JOIN with multiple conditions
SELECT o.id, o.name, c.title FROM organizations o, competitions c WHERE c.organization_id = o.id AND c.total_teams > 100 LIMIT 10;
-- EXPECT_ROWS: >= 1

-- Q41: Cross join with filter
SELECT a.id, b.id FROM (SELECT id FROM organizations ORDER BY id ASC LIMIT 4) AS a, (SELECT id FROM organizations ORDER BY id DESC LIMIT 4) AS b WHERE a.id != b.id LIMIT 15;
-- EXPECT_ROWS: >= 1

-- Q42: Three-way join
SELECT ct.id, c.title, t.name FROM competition_tags ct, competitions c, tags t WHERE ct.competition_id = c.id AND ct.tag_id = t.id LIMIT 10;
-- EXPECT_ROWS: >= 1

-- =============================================================================
-- SECTION 7: Complex WHERE Clauses
-- =============================================================================

-- Q43: WHERE with BETWEEN
SELECT * FROM (SELECT id, name FROM organizations WHERE id BETWEEN 100 AND 500 ORDER BY id LIMIT 10);
-- EXPECT_ROWS: 10

-- Q44: WHERE with multiple LIKE patterns using OR
SELECT * FROM (SELECT id, name FROM organizations WHERE name LIKE 'A%' OR name LIKE 'B%' OR name LIKE 'C%' LIMIT 20);
-- EXPECT_ROWS: >= 1

-- Q45: Complex AND/OR logic with parentheses
SELECT * FROM (SELECT id, name FROM organizations WHERE (id > 100 AND id < 500) OR (id > 3000 AND id < 3500) LIMIT 20);
-- EXPECT_ROWS: >= 1

-- Q46: NOT operator
SELECT * FROM (SELECT id, name FROM organizations WHERE NOT (id < 100) ORDER BY id ASC LIMIT 10);
-- EXPECT_ROWS: 10

-- Q47: WHERE with arithmetic comparison
SELECT * FROM (SELECT id, name, id * 2 AS doubled FROM organizations WHERE id * 2 > 1000 LIMIT 10);
-- EXPECT_ROWS: >= 1

-- Q48: Nested WHERE through subqueries
SELECT * FROM (SELECT * FROM (SELECT id, name, slug FROM organizations WHERE id > 50 LIMIT 100) AS sub WHERE name LIKE '%a%' LIMIT 20) AS final;
-- EXPECT_ROWS: >= 1

-- Q49: WHERE with IN list
SELECT * FROM (SELECT id, name FROM organizations WHERE id IN (2, 100, 500, 1000, 2000) LIMIT 10);
-- EXPECT_ROWS: >= 1

-- Q50: WHERE with NOT IN
SELECT * FROM (SELECT id, name FROM organizations WHERE id NOT IN (1, 2, 3, 4, 5) ORDER BY id ASC LIMIT 10);
-- EXPECT_ROWS: >= 1

-- =============================================================================
-- SECTION 8: Aggregate Functions
-- =============================================================================

-- Q51: COUNT(*)
SELECT * FROM (SELECT COUNT(*) as total FROM organizations);
-- EXPECT_ROWS: 1

-- Q52: Multiple aggregates
SELECT * FROM (SELECT COUNT(*) as cnt, MIN(id) as min_id, MAX(id) as max_id, SUM(id) as sum_id FROM organizations);
-- EXPECT_ROWS: 1

-- Q53: AVG function
SELECT * FROM (SELECT AVG(id) as avg_id FROM organizations);
-- EXPECT_ROWS: 1

-- Q54: COUNT with column (excludes NULLs)
SELECT * FROM (SELECT COUNT(organization_id) as non_null_count, COUNT(*) as total_count FROM competitions);
-- EXPECT_ROWS: 1

-- Q55: Aggregate on filtered data
SELECT * FROM (SELECT COUNT(*) as cnt, AVG(id) as avg_id FROM organizations WHERE id > 1000);
-- EXPECT_ROWS: 1

-- Q56: Nested aggregate
SELECT * FROM (SELECT MAX(cnt) as max_cnt FROM (SELECT creation_date, COUNT(*) as cnt FROM organizations GROUP BY creation_date LIMIT 50) AS sub);
-- EXPECT_ROWS: 1

-- Q57: SUM with expression
SELECT * FROM (SELECT SUM(id * 2) as doubled_sum FROM organizations WHERE id < 50);
-- EXPECT_ROWS: 1

-- Q58: Aggregate on competition rewards
SELECT * FROM (SELECT SUM(reward_quantity) as total_rewards, AVG(reward_quantity) as avg_reward FROM competitions WHERE reward_quantity > 0);
-- EXPECT_ROWS: 1

-- =============================================================================
-- SECTION 9: GROUP BY Operations
-- =============================================================================

-- Q59: Simple GROUP BY
SELECT * FROM (SELECT creation_date, COUNT(*) as cnt FROM organizations GROUP BY creation_date LIMIT 10);
-- EXPECT_ROWS: 10

-- Q60: GROUP BY with ORDER BY on aggregate
SELECT * FROM (SELECT creation_date, COUNT(*) as cnt FROM organizations GROUP BY creation_date ORDER BY cnt DESC LIMIT 5);
-- EXPECT_ROWS: 5

-- Q61: GROUP BY with multiple aggregates
SELECT * FROM (SELECT creation_date, COUNT(*) as cnt, MIN(id) as min_id, MAX(id) as max_id FROM organizations GROUP BY creation_date LIMIT 10);
-- EXPECT_ROWS: 10

-- Q62: GROUP BY with filter before grouping
SELECT * FROM (SELECT creation_date, COUNT(*) as cnt FROM organizations WHERE id > 100 GROUP BY creation_date LIMIT 10);
-- EXPECT_ROWS: >= 1

-- Q63: GROUP BY on competition data
SELECT * FROM (SELECT reward_type, COUNT(*) as cnt, SUM(reward_quantity) as total_reward FROM competitions WHERE reward_quantity > 0 GROUP BY reward_type LIMIT 10);
-- EXPECT_ROWS: >= 1

-- Q64: GROUP BY with text column
SELECT * FROM (SELECT reward_type, COUNT(*) as cnt FROM competitions GROUP BY reward_type ORDER BY cnt DESC LIMIT 5);
-- EXPECT_ROWS: >= 1

-- Q65: GROUP BY on expression (id % 100)
SELECT * FROM (SELECT id % 100 as bucket, COUNT(*) as cnt FROM organizations GROUP BY id % 100 ORDER BY cnt DESC LIMIT 10);
-- EXPECT_ROWS: 10

-- =============================================================================
-- SECTION 10: Complex Expressions
-- =============================================================================

-- Q66: Arithmetic expressions
SELECT * FROM (SELECT id, id * 2 as doubled, id + 100 as plus_100, id - 1 as minus_1, id / 2 as halved FROM organizations LIMIT 5);
-- EXPECT_ROWS: 5

-- Q67: String concatenation
SELECT * FROM (SELECT id, name || ' (' || slug || ')' as full_name FROM organizations LIMIT 5);
-- EXPECT_ROWS: 5

-- Q68: CASE with multiple WHEN clauses
SELECT * FROM (SELECT id, name, CASE WHEN id < 100 THEN 'tiny' WHEN id < 500 THEN 'small' WHEN id < 1000 THEN 'medium' WHEN id < 2000 THEN 'large' ELSE 'huge' END as size_category FROM organizations LIMIT 20);
-- EXPECT_ROWS: 20

-- Q69: Nested CASE expressions
SELECT * FROM (SELECT id, CASE WHEN id % 2 = 0 THEN CASE WHEN id < 1000 THEN 'even-small' ELSE 'even-large' END ELSE CASE WHEN id < 1000 THEN 'odd-small' ELSE 'odd-large' END END as category FROM organizations LIMIT 10);
-- EXPECT_ROWS: 10

-- Q70: COALESCE with multiple fallbacks
SELECT * FROM (SELECT id, COALESCE(description, name, 'Unknown') as display_text FROM organizations LIMIT 10);
-- EXPECT_ROWS: 10

-- Q71: Modulo operations
SELECT * FROM (SELECT id, id % 10 as mod_10, id % 100 as mod_100 FROM organizations ORDER BY id % 10, id LIMIT 20);
-- EXPECT_ROWS: 20

-- Q72: Complex combined expression with fizzbuzz
SELECT * FROM (SELECT id, name, (id * 100) / (id + 1) as ratio, CASE WHEN id % 3 = 0 AND id % 5 = 0 THEN 'fizzbuzz' WHEN id % 3 = 0 THEN 'fizz' WHEN id % 5 = 0 THEN 'buzz' ELSE 'normal' END as fizzbuzz FROM organizations LIMIT 15);
-- EXPECT_ROWS: 15

-- =============================================================================
-- SECTION 11: ORDER BY Variations
-- =============================================================================

-- Q73: ORDER BY multiple columns
SELECT * FROM (SELECT id, name, creation_date FROM organizations ORDER BY creation_date DESC, id ASC LIMIT 10);
-- EXPECT_ROWS: 10

-- Q74: ORDER BY expression
SELECT * FROM (SELECT id, name, id % 7 as mod_7 FROM organizations ORDER BY id % 7 DESC, id ASC LIMIT 15);
-- EXPECT_ROWS: 15

-- Q75: ORDER BY with CASE expression
SELECT * FROM (SELECT id, name, CASE WHEN id < 500 THEN 1 ELSE 2 END as priority FROM organizations ORDER BY CASE WHEN id < 500 THEN 1 ELSE 2 END, id LIMIT 10);
-- EXPECT_ROWS: 10

-- Q76: ORDER BY on outer query after subquery
SELECT * FROM (SELECT id, name FROM organizations LIMIT 50) AS sub ORDER BY id DESC LIMIT 10;
-- EXPECT_ROWS: 10

-- Q77: ORDER BY text column ASC
SELECT * FROM (SELECT id, name FROM organizations ORDER BY name ASC LIMIT 10);
-- EXPECT_ROWS: 10

-- Q78: ORDER BY text column DESC
SELECT * FROM (SELECT id, name FROM organizations ORDER BY name DESC LIMIT 10);
-- EXPECT_ROWS: 10

-- Q79: ORDER BY NULL handling
SELECT * FROM (SELECT id, organization_id FROM competitions ORDER BY organization_id ASC LIMIT 10);
-- EXPECT_ROWS: 10

-- =============================================================================
-- SECTION 12: LIMIT and OFFSET
-- =============================================================================

-- Q80: Basic OFFSET
SELECT * FROM (SELECT id, name FROM organizations LIMIT 5 OFFSET 10);
-- EXPECT_ROWS: 5

-- Q81: ORDER BY with OFFSET
SELECT * FROM (SELECT id, name FROM organizations ORDER BY id ASC LIMIT 5 OFFSET 20);
-- EXPECT_ROWS: 5

-- Q82: Chained LIMIT/OFFSET through subqueries
SELECT * FROM (SELECT * FROM (SELECT id, name FROM organizations ORDER BY id ASC LIMIT 100 OFFSET 50) AS sub LIMIT 20 OFFSET 10) AS final;
-- EXPECT_ROWS: 20

-- Q83: OFFSET with ORDER BY DESC
SELECT * FROM (SELECT id, name FROM organizations ORDER BY id DESC LIMIT 10 OFFSET 5);
-- EXPECT_ROWS: 10

-- Q84: Large OFFSET near table boundary
SELECT * FROM (SELECT id FROM organizations ORDER BY id DESC LIMIT 50 OFFSET 700);
-- EXPECT_ROWS: >= 1

-- =============================================================================
-- SECTION 13: DISTINCT Operations
-- =============================================================================

-- Q85: DISTINCT on single column
SELECT * FROM (SELECT DISTINCT creation_date FROM organizations LIMIT 20);
-- EXPECT_ROWS: 20

-- Q86: DISTINCT with ORDER BY
SELECT * FROM (SELECT DISTINCT creation_date FROM organizations ORDER BY creation_date DESC LIMIT 10);
-- EXPECT_ROWS: 10

-- Q87: DISTINCT on expression result
SELECT * FROM (SELECT DISTINCT id % 10 as mod_result FROM organizations LIMIT 10);
-- EXPECT_ROWS: 10

-- Q88: DISTINCT with multiple columns
SELECT * FROM (SELECT DISTINCT reward_type, has_kernels FROM competitions LIMIT 10);
-- EXPECT_ROWS: >= 1

-- =============================================================================
-- SECTION 14: Window Functions
-- =============================================================================

-- Q89: ROW_NUMBER basic
SELECT * FROM (SELECT id, name, ROW_NUMBER() OVER (ORDER BY id) as rn FROM organizations LIMIT 10);
-- EXPECT_ROWS: 10

-- Q90: ROW_NUMBER DESC
SELECT * FROM (SELECT id, name, ROW_NUMBER() OVER (ORDER BY id DESC) as rn FROM organizations LIMIT 10);
-- EXPECT_ROWS: 10

-- Q91: Window function on competition data
SELECT * FROM (SELECT id, title, ROW_NUMBER() OVER (ORDER BY total_teams DESC) as rank FROM competitions LIMIT 10);
-- EXPECT_ROWS: 10

-- Q92: Nested subquery with window function
SELECT * FROM (SELECT * FROM (SELECT id, ROW_NUMBER() OVER (ORDER BY id) as rn FROM organizations LIMIT 50) AS sub LIMIT 20);
-- EXPECT_ROWS: 20

-- Q93: Window function with filter on result
SELECT * FROM (SELECT * FROM (SELECT id, ROW_NUMBER() OVER (ORDER BY id) as rn FROM organizations LIMIT 50) AS sub WHERE rn <= 10);
-- EXPECT_ROWS: 10

-- =============================================================================
-- SECTION 15: NULL Handling
-- =============================================================================

-- Q94: Filter on NULL
SELECT * FROM (SELECT id, title, organization_id FROM competitions WHERE organization_id IS NULL LIMIT 10);
-- EXPECT_ROWS: >= 1

-- Q95: Filter on NOT NULL
SELECT * FROM (SELECT id, title, organization_id FROM competitions WHERE organization_id IS NOT NULL LIMIT 10);
-- EXPECT_ROWS: >= 1

-- Q96: COALESCE with NULL
SELECT * FROM (SELECT id, title, COALESCE(organization_id, -1) as org_id FROM competitions LIMIT 10);
-- EXPECT_ROWS: 10

-- Q97: NULL in ORDER BY
SELECT * FROM (SELECT id, organization_id FROM competitions ORDER BY organization_id DESC LIMIT 20);
-- EXPECT_ROWS: 20

-- Q98: CASE with NULL check
SELECT * FROM (SELECT id, title, CASE WHEN organization_id IS NULL THEN 'No Org' ELSE 'Has Org' END as org_status FROM competitions LIMIT 10);
-- EXPECT_ROWS: 10

-- =============================================================================
-- SECTION 16: String Operations
-- =============================================================================

-- Q99: LIKE with wildcard at start
SELECT * FROM (SELECT id, name FROM organizations WHERE name LIKE '%a%' LIMIT 10);
-- EXPECT_ROWS: >= 1

-- Q100: LIKE with wildcard at end
SELECT * FROM (SELECT id, name FROM organizations WHERE name LIKE 'A%' LIMIT 10);
-- EXPECT_ROWS: >= 1

-- Q101: LIKE case sensitivity test
SELECT * FROM (SELECT id, name FROM organizations WHERE name LIKE '%THE%' OR name LIKE '%the%' LIMIT 10);
-- EXPECT_ROWS: >= 1

-- Q102: Multiple string conditions
SELECT * FROM (SELECT id, name, slug FROM organizations WHERE name LIKE '%Data%' OR slug LIKE '%data%' LIMIT 10);
-- EXPECT_ROWS: >= 1

-- Q103: String concatenation with NULL handling
SELECT * FROM (SELECT id, COALESCE(name, '') || ' - ' || COALESCE(slug, '') as combined FROM organizations LIMIT 10);
-- EXPECT_ROWS: 10

-- =============================================================================
-- SECTION 17: Edge Cases
-- =============================================================================

-- Q104: Empty result set
SELECT * FROM (SELECT id, name FROM organizations WHERE id < 0 LIMIT 10);
-- EXPECT_ROWS: 0

-- Q105: Single row result
SELECT * FROM (SELECT id, name FROM organizations ORDER BY id DESC LIMIT 1);
-- EXPECT_ROWS: 1

-- Q106: Zero LIMIT
SELECT * FROM (SELECT id FROM organizations LIMIT 0);
-- EXPECT_ROWS: 0

-- Q107: LIMIT larger than result set with filter
SELECT COUNT(*) FROM (SELECT id FROM organizations WHERE id < 50 LIMIT 1000);
-- EXPECT_ROWS: 1

-- Q108: Multiple identical subqueries (caching test)
SELECT * FROM (SELECT id FROM organizations ORDER BY id ASC LIMIT 5) AS a, (SELECT id FROM organizations ORDER BY id ASC LIMIT 5) AS b WHERE a.id = b.id;
-- EXPECT_ROWS: 5

-- =============================================================================
-- SECTION 18: Complex Combined Scenarios
-- =============================================================================

-- Q109: Filter -> Aggregate -> Filter pattern
SELECT * FROM (SELECT * FROM (SELECT creation_date, COUNT(*) as cnt FROM organizations GROUP BY creation_date) AS grouped WHERE cnt >= 1 LIMIT 10);
-- EXPECT_ROWS: >= 1

-- Q110: Multiple ORDER BY through nested queries
SELECT * FROM (SELECT * FROM (SELECT * FROM (SELECT id, name FROM organizations ORDER BY name ASC LIMIT 100) AS s1 ORDER BY id DESC LIMIT 50) AS s2 ORDER BY name DESC LIMIT 20) AS s3;
-- EXPECT_ROWS: 20

-- Q111: Aggregate of aggregate through nesting
SELECT * FROM (SELECT SUM(cnt) as total FROM (SELECT creation_date, COUNT(*) as cnt FROM organizations GROUP BY creation_date LIMIT 50) AS sub);
-- EXPECT_ROWS: 1

-- Q112: Subquery in FROM with expression columns used in outer filter
SELECT * FROM (SELECT * FROM (SELECT id, name, id * 10 as score FROM organizations) AS scored WHERE score > 5000 AND score < 20000 LIMIT 10);
-- EXPECT_ROWS: >= 1

-- Q113: Complex nested with multiple operations
SELECT final.id, final.name FROM (
    SELECT * FROM (
        SELECT * FROM (
            SELECT id, name, creation_date FROM organizations
            WHERE id > 10
            ORDER BY id DESC
            LIMIT 500
        ) AS l1
        WHERE id < 3500
        ORDER BY creation_date DESC
        LIMIT 200
    ) AS l2
    WHERE name LIKE '%a%'
    ORDER BY id ASC
    LIMIT 50
) AS final
ORDER BY id DESC
LIMIT 10;
-- EXPECT_ROWS: >= 1

-- Q114: Large result set aggregation
SELECT * FROM (SELECT COUNT(*) as total, AVG(id) as avg_id, SUM(id) as sum_id FROM organizations);
-- EXPECT_ROWS: 1

-- =============================================================================
-- SECTION 19: Numeric Operations
-- =============================================================================

-- Q115: Division
SELECT * FROM (SELECT id, id / 2 as half, id / 3 as third FROM organizations WHERE id > 0 LIMIT 10);
-- EXPECT_ROWS: 10

-- Q116: Large number arithmetic
SELECT * FROM (SELECT id, id * 1000000 as big_num FROM organizations LIMIT 5);
-- EXPECT_ROWS: 5

-- Q117: Negative results
SELECT * FROM (SELECT id, 100 - id as diff FROM organizations WHERE id > 50 LIMIT 10);
-- EXPECT_ROWS: 10

-- Q118: Comparison with computed values
SELECT * FROM (SELECT id, name FROM organizations WHERE id * 2 > 1000 AND id * 2 < 2000 LIMIT 10);
-- EXPECT_ROWS: >= 1

-- Q119: Modulo with different divisors
SELECT * FROM (SELECT id, id % 2 as mod2, id % 3 as mod3, id % 5 as mod5, id % 7 as mod7 FROM organizations LIMIT 10);
-- EXPECT_ROWS: 10

-- =============================================================================
-- SECTION 20: Subquery Scalar Operations
-- =============================================================================

-- Q120: Scalar subquery in SELECT (if supported)
SELECT id, name, (SELECT MAX(id) FROM organizations) as max_id FROM organizations LIMIT 5;
-- EXPECT_ROWS: 5

-- Q121: Scalar subquery in WHERE (if supported)
SELECT * FROM organizations WHERE id = (SELECT MIN(id) FROM organizations);
-- EXPECT_ROWS: 1

-- Q122: Subquery with IN from another table
SELECT * FROM organizations WHERE id IN (SELECT organization_id FROM user_organizations) LIMIT 10;
-- EXPECT_ROWS: >= 1

-- Q123: Correlated subquery pattern (if supported)
SELECT id, name FROM organizations o WHERE id > (SELECT AVG(id) FROM organizations) LIMIT 10;
-- EXPECT_ROWS: >= 1

-- Q124: Complex scalar subquery
SELECT id, name, id - (SELECT MIN(id) FROM organizations) as offset_from_min FROM organizations LIMIT 10;
-- EXPECT_ROWS: 10

-- =============================================================================
-- SECTION 21: JOIN Inside Subquery (Known limitation test)
-- =============================================================================

-- Q125: JOIN wrapped in subquery
SELECT * FROM (SELECT o.id, o.name, uo.user_id FROM organizations o, user_organizations uo WHERE o.id = uo.organization_id LIMIT 20);
-- EXPECT_ROWS: >= 1

-- Q126: Aggregate JOIN in subquery
SELECT * FROM (SELECT o.id, o.name, COUNT(*) as cnt FROM organizations o, user_organizations uo WHERE o.id = uo.organization_id GROUP BY o.id, o.name LIMIT 10);
-- EXPECT_ROWS: >= 1

-- Q127: Complex JOIN in subquery with expressions
SELECT * FROM (SELECT o.id, o.name, c.title, o.id * 100 as score FROM organizations o, competitions c WHERE c.organization_id = o.id ORDER BY score DESC LIMIT 10);
-- EXPECT_ROWS: >= 1

-- =============================================================================
-- SECTION 22: UNION Operations (if supported)
-- =============================================================================

-- Q128: Simple UNION
SELECT id, name FROM organizations WHERE id < 10 UNION SELECT id, name FROM organizations WHERE id > 3870 LIMIT 20;
-- EXPECT_ROWS: >= 1

-- Q129: UNION ALL
SELECT id, name FROM organizations WHERE id < 10 UNION ALL SELECT id, name FROM organizations WHERE id < 10 LIMIT 20;
-- EXPECT_ROWS: >= 1

-- Q130: UNION with different filters
SELECT id, name FROM organizations WHERE name LIKE 'A%' UNION SELECT id, name FROM organizations WHERE name LIKE 'Z%' LIMIT 20;
-- EXPECT_ROWS: >= 1

-- =============================================================================
-- SECTION 23: String Functions
-- =============================================================================

-- Q131: UPPER function
SELECT * FROM (SELECT id, UPPER(name) as upper_name FROM organizations LIMIT 5);
-- EXPECT_ROWS: 5

-- Q132: LOWER function
SELECT * FROM (SELECT id, LOWER(name) as lower_name FROM organizations LIMIT 5);
-- EXPECT_ROWS: 5

-- Q133: LENGTH function
SELECT * FROM (SELECT id, name, LENGTH(name) as name_len FROM organizations LIMIT 5);
-- EXPECT_ROWS: 5

-- Q134: CHAR_LENGTH function
SELECT * FROM (SELECT id, name, CHAR_LENGTH(name) as char_len FROM organizations LIMIT 5);
-- EXPECT_ROWS: 5

-- Q135: CONCAT function
SELECT * FROM (SELECT id, CONCAT(name, '-', slug) as combined FROM organizations LIMIT 5);
-- EXPECT_ROWS: 5

-- Q136: CONCAT_WS function (with separator)
SELECT * FROM (SELECT id, CONCAT_WS(' | ', name, slug) as combined FROM organizations LIMIT 5);
-- EXPECT_ROWS: 5

-- Q137: SUBSTR/SUBSTRING function
SELECT * FROM (SELECT id, name, SUBSTR(name, 1, 5) as short_name FROM organizations LIMIT 5);
-- EXPECT_ROWS: 5

-- Q138: LEFT function
SELECT * FROM (SELECT id, name, LEFT(name, 3) as left_chars FROM organizations LIMIT 5);
-- EXPECT_ROWS: 5

-- Q139: RIGHT function
SELECT * FROM (SELECT id, name, RIGHT(name, 3) as right_chars FROM organizations LIMIT 5);
-- EXPECT_ROWS: 5

-- Q140: TRIM function
SELECT * FROM (SELECT id, TRIM(name) as trimmed FROM organizations LIMIT 5);
-- EXPECT_ROWS: 5

-- Q141: LTRIM function
SELECT * FROM (SELECT id, LTRIM(name) as left_trimmed FROM organizations LIMIT 5);
-- EXPECT_ROWS: 5

-- Q142: RTRIM function
SELECT * FROM (SELECT id, RTRIM(name) as right_trimmed FROM organizations LIMIT 5);
-- EXPECT_ROWS: 5

-- Q143: REPLACE function
SELECT * FROM (SELECT id, name, REPLACE(name, 'a', 'X') as replaced FROM organizations WHERE name LIKE '%a%' LIMIT 5);
-- EXPECT_ROWS: >= 1

-- Q144: REVERSE function
SELECT * FROM (SELECT id, name, REVERSE(name) as reversed FROM organizations LIMIT 5);
-- EXPECT_ROWS: 5

-- Q145: REPEAT function
SELECT * FROM (SELECT id, REPEAT('*', 5) as stars FROM organizations LIMIT 3);
-- EXPECT_ROWS: 3

-- Q146: SPACE function
SELECT * FROM (SELECT id, CONCAT(name, SPACE(5), slug) as spaced FROM organizations LIMIT 3);
-- EXPECT_ROWS: 3

-- Q147: LPAD function
SELECT * FROM (SELECT id, LPAD(name, 20, '*') as padded FROM organizations LIMIT 5);
-- EXPECT_ROWS: 5

-- Q148: RPAD function
SELECT * FROM (SELECT id, RPAD(name, 20, '*') as padded FROM organizations LIMIT 5);
-- EXPECT_ROWS: 5

-- Q149: ASCII function
SELECT * FROM (SELECT id, name, ASCII(name) as first_ascii FROM organizations LIMIT 5);
-- EXPECT_ROWS: 5

-- Q150: INSTR function
SELECT * FROM (SELECT id, name, INSTR(name, 'a') as pos FROM organizations LIMIT 5);
-- EXPECT_ROWS: 5

-- Q151: LOCATE function
SELECT * FROM (SELECT id, name, LOCATE('a', name) as pos FROM organizations LIMIT 5);
-- EXPECT_ROWS: 5

-- Q152: SUBSTRING_INDEX function
SELECT * FROM (SELECT id, slug, SUBSTRING_INDEX(slug, '-', 1) as first_part FROM organizations WHERE slug LIKE '%-%' LIMIT 5);
-- EXPECT_ROWS: >= 1

-- Q153: STRCMP function
SELECT * FROM (SELECT id, name, STRCMP(name, 'AAA') as cmp FROM organizations LIMIT 5);
-- EXPECT_ROWS: 5

-- Q154: FORMAT function
SELECT * FROM (SELECT id, FORMAT(id * 1000.5, 2) as formatted FROM organizations LIMIT 5);
-- EXPECT_ROWS: 5

-- Q155: INSERT function (string insertion)
SELECT * FROM (SELECT id, name, INSERT(name, 2, 2, 'XX') as modified FROM organizations LIMIT 5);
-- EXPECT_ROWS: 5

-- =============================================================================
-- SECTION 24: Numeric Functions
-- =============================================================================

-- Q156: ABS function
SELECT * FROM (SELECT id, ABS(id - 500) as abs_diff FROM organizations LIMIT 5);
-- EXPECT_ROWS: 5

-- Q157: SIGN function
SELECT * FROM (SELECT id, SIGN(id - 500) as sign_val FROM organizations LIMIT 10);
-- EXPECT_ROWS: 10

-- Q158: CEIL/CEILING function
SELECT * FROM (SELECT id, id / 3.0 as division, CEIL(id / 3.0) as ceiled FROM organizations LIMIT 5);
-- EXPECT_ROWS: 5

-- Q159: FLOOR function
SELECT * FROM (SELECT id, id / 3.0 as division, FLOOR(id / 3.0) as floored FROM organizations LIMIT 5);
-- EXPECT_ROWS: 5

-- Q160: ROUND function
SELECT * FROM (SELECT id, id / 7.0 as division, ROUND(id / 7.0, 2) as rounded FROM organizations LIMIT 5);
-- EXPECT_ROWS: 5

-- Q161: TRUNCATE function
SELECT * FROM (SELECT id, id / 7.0 as division, TRUNCATE(id / 7.0, 2) as truncated FROM organizations LIMIT 5);
-- EXPECT_ROWS: 5

-- Q162: MOD function
SELECT * FROM (SELECT id, MOD(id, 7) as mod_7 FROM organizations LIMIT 10);
-- EXPECT_ROWS: 10

-- Q163: DIV function (integer division)
SELECT * FROM (SELECT id, DIV(id, 7) as int_div FROM organizations LIMIT 10);
-- EXPECT_ROWS: 10

-- Q164: SQRT function
SELECT * FROM (SELECT id, SQRT(id) as sqrt_id FROM organizations WHERE id > 0 LIMIT 5);
-- EXPECT_ROWS: 5

-- Q165: POWER/POW function
SELECT * FROM (SELECT id, POWER(id, 2) as squared FROM organizations WHERE id < 100 LIMIT 5);
-- EXPECT_ROWS: 5

-- Q166: EXP function
SELECT * FROM (SELECT id, EXP(id / 100.0) as exp_val FROM organizations WHERE id < 100 LIMIT 5);
-- EXPECT_ROWS: 5

-- Q167: LN function (natural log)
SELECT * FROM (SELECT id, LN(id) as ln_val FROM organizations WHERE id > 0 LIMIT 5);
-- EXPECT_ROWS: 5

-- Q168: LOG function (base 10)
SELECT * FROM (SELECT id, LOG(id) as log_val FROM organizations WHERE id > 0 LIMIT 5);
-- EXPECT_ROWS: 5

-- Q169: LOG10 function
SELECT * FROM (SELECT id, LOG10(id) as log10_val FROM organizations WHERE id > 0 LIMIT 5);
-- EXPECT_ROWS: 5

-- Q170: LOG2 function
SELECT * FROM (SELECT id, LOG2(id) as log2_val FROM organizations WHERE id > 0 LIMIT 5);
-- EXPECT_ROWS: 5

-- Q171: SIN function
SELECT * FROM (SELECT id, SIN(id) as sin_val FROM organizations LIMIT 5);
-- EXPECT_ROWS: 5

-- Q172: COS function
SELECT * FROM (SELECT id, COS(id) as cos_val FROM organizations LIMIT 5);
-- EXPECT_ROWS: 5

-- Q173: TAN function
SELECT * FROM (SELECT id, TAN(id) as tan_val FROM organizations LIMIT 5);
-- EXPECT_ROWS: 5

-- Q174: ASIN function
SELECT * FROM (SELECT id, ASIN(id / 1000.0) as asin_val FROM organizations WHERE id < 1000 LIMIT 5);
-- EXPECT_ROWS: 5

-- Q175: ACOS function
SELECT * FROM (SELECT id, ACOS(id / 1000.0) as acos_val FROM organizations WHERE id < 1000 LIMIT 5);
-- EXPECT_ROWS: 5

-- Q176: ATAN function
SELECT * FROM (SELECT id, ATAN(id) as atan_val FROM organizations LIMIT 5);
-- EXPECT_ROWS: 5

-- Q177: ATAN2 function
SELECT * FROM (SELECT id, ATAN2(id, 100) as atan2_val FROM organizations LIMIT 5);
-- EXPECT_ROWS: 5

-- Q178: COT function
SELECT * FROM (SELECT id, COT(id + 0.1) as cot_val FROM organizations WHERE id > 0 LIMIT 5);
-- EXPECT_ROWS: 5

-- Q179: DEGREES function
SELECT * FROM (SELECT id, DEGREES(id / 100.0) as degrees_val FROM organizations LIMIT 5);
-- EXPECT_ROWS: 5

-- Q180: RADIANS function
SELECT * FROM (SELECT id, RADIANS(id) as radians_val FROM organizations LIMIT 5);
-- EXPECT_ROWS: 5

-- Q181: PI function
SELECT * FROM (SELECT id, PI() as pi_val FROM organizations LIMIT 3);
-- EXPECT_ROWS: 3

-- Q182: RAND function
SELECT * FROM (SELECT id, RAND() as random_val FROM organizations LIMIT 5);
-- EXPECT_ROWS: 5

-- Q183: GREATEST function
SELECT * FROM (SELECT id, GREATEST(id, 100, 500) as greatest_val FROM organizations LIMIT 10);
-- EXPECT_ROWS: 10

-- Q184: LEAST function
SELECT * FROM (SELECT id, LEAST(id, 100, 500) as least_val FROM organizations LIMIT 10);
-- EXPECT_ROWS: 10

-- =============================================================================
-- SECTION 25: DateTime Functions
-- =============================================================================

-- Q185: NOW function
SELECT * FROM (SELECT id, NOW() as current_time FROM organizations LIMIT 3);
-- EXPECT_ROWS: 3

-- Q186: CURRENT_DATE function
SELECT * FROM (SELECT id, CURRENT_DATE() as today FROM organizations LIMIT 3);
-- EXPECT_ROWS: 3

-- Q187: CURRENT_TIME function
SELECT * FROM (SELECT id, CURRENT_TIME() as time_now FROM organizations LIMIT 3);
-- EXPECT_ROWS: 3

-- Q188: CURRENT_TIMESTAMP function
SELECT * FROM (SELECT id, CURRENT_TIMESTAMP() as ts FROM organizations LIMIT 3);
-- EXPECT_ROWS: 3

-- Q189: DATE function
SELECT * FROM (SELECT id, creation_date, DATE(creation_date) as date_part FROM organizations LIMIT 5);
-- EXPECT_ROWS: 5

-- Q190: TIME function
SELECT * FROM (SELECT id, enabled_date, TIME(enabled_date) as time_part FROM competitions LIMIT 5);
-- EXPECT_ROWS: 5

-- Q191: YEAR function
SELECT * FROM (SELECT id, creation_date, YEAR(creation_date) as year_val FROM organizations LIMIT 5);
-- EXPECT_ROWS: 5

-- Q192: MONTH function
SELECT * FROM (SELECT id, creation_date, MONTH(creation_date) as month_val FROM organizations LIMIT 5);
-- EXPECT_ROWS: 5

-- Q193: DAY/DAYOFMONTH function
SELECT * FROM (SELECT id, creation_date, DAY(creation_date) as day_val FROM organizations LIMIT 5);
-- EXPECT_ROWS: 5

-- Q194: HOUR function
SELECT * FROM (SELECT id, enabled_date, HOUR(enabled_date) as hour_val FROM competitions LIMIT 5);
-- EXPECT_ROWS: 5

-- Q195: MINUTE function
SELECT * FROM (SELECT id, enabled_date, MINUTE(enabled_date) as minute_val FROM competitions LIMIT 5);
-- EXPECT_ROWS: 5

-- Q196: SECOND function
SELECT * FROM (SELECT id, enabled_date, SECOND(enabled_date) as second_val FROM competitions LIMIT 5);
-- EXPECT_ROWS: 5

-- Q197: DAYNAME function
SELECT * FROM (SELECT id, creation_date, DAYNAME(creation_date) as day_name FROM organizations LIMIT 5);
-- EXPECT_ROWS: 5

-- Q198: MONTHNAME function
SELECT * FROM (SELECT id, creation_date, MONTHNAME(creation_date) as month_name FROM organizations LIMIT 5);
-- EXPECT_ROWS: 5

-- Q199: DAYOFWEEK function
SELECT * FROM (SELECT id, creation_date, DAYOFWEEK(creation_date) as dow FROM organizations LIMIT 5);
-- EXPECT_ROWS: 5

-- Q200: DAYOFYEAR function
SELECT * FROM (SELECT id, creation_date, DAYOFYEAR(creation_date) as doy FROM organizations LIMIT 5);
-- EXPECT_ROWS: 5

-- Q201: WEEKDAY function
SELECT * FROM (SELECT id, creation_date, WEEKDAY(creation_date) as weekday_val FROM organizations LIMIT 5);
-- EXPECT_ROWS: 5

-- Q202: WEEK function
SELECT * FROM (SELECT id, creation_date, WEEK(creation_date) as week_val FROM organizations LIMIT 5);
-- EXPECT_ROWS: 5

-- Q203: QUARTER function
SELECT * FROM (SELECT id, creation_date, QUARTER(creation_date) as quarter_val FROM organizations LIMIT 5);
-- EXPECT_ROWS: 5

-- Q204: YEARWEEK function
SELECT * FROM (SELECT id, creation_date, YEARWEEK(creation_date) as yearweek_val FROM organizations LIMIT 5);
-- EXPECT_ROWS: 5

-- Q205: LAST_DAY function
SELECT * FROM (SELECT id, creation_date, LAST_DAY(creation_date) as last_day_val FROM organizations LIMIT 5);
-- EXPECT_ROWS: 5

-- Q206: DATEDIFF function
SELECT * FROM (SELECT id, creation_date, DATEDIFF(NOW(), creation_date) as days_ago FROM organizations LIMIT 5);
-- EXPECT_ROWS: 5

-- Q207: DATE_ADD function
SELECT * FROM (SELECT id, creation_date, DATE_ADD(creation_date, 30) as plus_30_days FROM organizations LIMIT 5);
-- EXPECT_ROWS: 5

-- Q208: DATE_SUB function
SELECT * FROM (SELECT id, creation_date, DATE_SUB(creation_date, 30) as minus_30_days FROM organizations LIMIT 5);
-- EXPECT_ROWS: 5

-- Q209: TO_DAYS function
SELECT * FROM (SELECT id, creation_date, TO_DAYS(creation_date) as total_days FROM organizations LIMIT 5);
-- EXPECT_ROWS: 5

-- Q210: FROM_DAYS function
SELECT * FROM (SELECT id, FROM_DAYS(738000) as date_from_days FROM organizations LIMIT 3);
-- EXPECT_ROWS: 3

-- Q211: TIME_TO_SEC function
SELECT * FROM (SELECT id, TIME_TO_SEC('10:30:00') as secs FROM organizations LIMIT 3);
-- EXPECT_ROWS: 3

-- Q212: SEC_TO_TIME function
SELECT * FROM (SELECT id, SEC_TO_TIME(3661) as time_val FROM organizations LIMIT 3);
-- EXPECT_ROWS: 3

-- Q213: MAKEDATE function
SELECT * FROM (SELECT id, MAKEDATE(2024, 100) as date_val FROM organizations LIMIT 3);
-- EXPECT_ROWS: 3

-- Q214: MAKETIME function
SELECT * FROM (SELECT id, MAKETIME(10, 30, 45) as time_val FROM organizations LIMIT 3);
-- EXPECT_ROWS: 3

-- Q215: TIMESTAMP function
SELECT * FROM (SELECT id, TIMESTAMP(creation_date) as ts FROM organizations LIMIT 5);
-- EXPECT_ROWS: 5

-- =============================================================================
-- SECTION 26: System/Control Functions
-- =============================================================================

-- Q216: VERSION function
SELECT VERSION();
-- EXPECT_ROWS: 1

-- Q217: TYPEOF function
SELECT * FROM (SELECT id, TYPEOF(id) as id_type, TYPEOF(name) as name_type FROM organizations LIMIT 3);
-- EXPECT_ROWS: 3

-- Q218: IF function
SELECT * FROM (SELECT id, IF(id > 500, 'big', 'small') as size FROM organizations LIMIT 10);
-- EXPECT_ROWS: 10

-- Q219: IFNULL function
SELECT * FROM (SELECT id, IFNULL(organization_id, -1) as org FROM competitions LIMIT 10);
-- EXPECT_ROWS: 10

-- Q220: NULLIF function
SELECT * FROM (SELECT id, NULLIF(id, 1) as nullified FROM organizations LIMIT 5);
-- EXPECT_ROWS: 5

-- Q221: ISNULL function
SELECT * FROM (SELECT id, ISNULL(organization_id) as is_null FROM competitions LIMIT 10);
-- EXPECT_ROWS: 10

-- Q222: BIN function
SELECT * FROM (SELECT id, BIN(id) as binary_val FROM organizations WHERE id < 20 LIMIT 10);
-- EXPECT_ROWS: >= 1

-- Q223: CONV function
SELECT * FROM (SELECT id, CONV(id, 10, 16) as hex_val FROM organizations WHERE id < 100 LIMIT 10);
-- EXPECT_ROWS: >= 1

-- =============================================================================
-- SECTION 27: Combined Function Tests
-- =============================================================================

-- Q224: String + Numeric combined
SELECT * FROM (SELECT id, name, LENGTH(name) as len, ROUND(LENGTH(name) / 2.0, 1) as half_len FROM organizations LIMIT 5);
-- EXPECT_ROWS: 5

-- Q225: Date extraction with filter
SELECT * FROM (SELECT id, creation_date, YEAR(creation_date) as y, MONTH(creation_date) as m FROM organizations WHERE YEAR(creation_date) >= 2020 LIMIT 10);
-- EXPECT_ROWS: >= 0

-- Q226: Nested function calls
SELECT * FROM (SELECT id, UPPER(REVERSE(LEFT(name, 5))) as transformed FROM organizations LIMIT 5);
-- EXPECT_ROWS: 5

-- Q227: Math chain
SELECT * FROM (SELECT id, ROUND(SQRT(ABS(id - 500)), 2) as chain_result FROM organizations LIMIT 5);
-- EXPECT_ROWS: 5

-- Q228: CASE with functions
SELECT * FROM (SELECT id, name, CASE WHEN LENGTH(name) > 10 THEN 'long' WHEN LENGTH(name) > 5 THEN 'medium' ELSE 'short' END as name_length_cat FROM organizations LIMIT 10);
-- EXPECT_ROWS: 10

-- Q229: Aggregate with function
SELECT * FROM (SELECT AVG(LENGTH(name)) as avg_name_len, MAX(LENGTH(name)) as max_name_len FROM organizations);
-- EXPECT_ROWS: 1

-- Q230: GROUP BY with function
SELECT * FROM (SELECT YEAR(creation_date) as year, COUNT(*) as cnt FROM organizations GROUP BY YEAR(creation_date) ORDER BY year DESC LIMIT 10);
-- EXPECT_ROWS: >= 1

-- Q231: Complex date calculation
SELECT * FROM (SELECT id, creation_date, DATEDIFF(NOW(), creation_date) as days_old, ROUND(DATEDIFF(NOW(), creation_date) / 365.0, 1) as years_old FROM organizations LIMIT 5);
-- EXPECT_ROWS: 5

-- Q232: String formatting chain
SELECT * FROM (SELECT id, CONCAT(UPPER(LEFT(name, 1)), LOWER(SUBSTR(name, 2))) as title_case FROM organizations LIMIT 5);
-- EXPECT_ROWS: 5

-- Q233: Trig calculation
SELECT * FROM (SELECT id, ROUND(SIN(RADIANS(id % 360)), 4) as sin_degrees FROM organizations LIMIT 10);
-- EXPECT_ROWS: 10

-- Q234: Multiple COALESCE with functions
SELECT * FROM (SELECT id, COALESCE(NULLIF(UPPER(name), ''), 'UNKNOWN') as safe_name FROM organizations LIMIT 5);
-- EXPECT_ROWS: 5

-- Q235: Filter with function result
SELECT * FROM (SELECT id, name, LENGTH(name) as len FROM organizations WHERE LENGTH(name) > 10 LIMIT 10);
-- EXPECT_ROWS: >= 1

-- Q236: ORDER BY function result
SELECT * FROM (SELECT id, name FROM organizations ORDER BY LENGTH(name) DESC LIMIT 10);
-- EXPECT_ROWS: 10

-- Q237: Window function with calculation
SELECT * FROM (SELECT id, name, SQRT(id) as sqrt_id, ROW_NUMBER() OVER (ORDER BY SQRT(id) DESC) as rn FROM organizations LIMIT 10);
-- EXPECT_ROWS: 10

-- Q238: GREATEST/LEAST with expressions
SELECT * FROM (SELECT id, GREATEST(id % 10, id % 7, id % 5) as max_mod, LEAST(id % 10, id % 7, id % 5) as min_mod FROM organizations LIMIT 10);
-- EXPECT_ROWS: 10

-- Q239: Complex string manipulation
SELECT * FROM (SELECT id, name, slug, CONCAT(LEFT(name, 3), '...', RIGHT(slug, 3)) as abbrev FROM organizations WHERE LENGTH(name) > 5 AND LENGTH(slug) > 5 LIMIT 5);
-- EXPECT_ROWS: >= 1

-- Q240: Date parts combined
SELECT * FROM (SELECT id, creation_date, CONCAT(YEAR(creation_date), '-Q', QUARTER(creation_date)) as year_quarter FROM organizations LIMIT 5);
-- EXPECT_ROWS: 5

-- =============================================================================
-- SECTION 28: DateTime Functions - Extended Tests
-- =============================================================================

-- Q241: CURDATE function (alias for CURRENT_DATE)
SELECT * FROM (SELECT id, CURDATE() as today FROM organizations LIMIT 3);
-- EXPECT_ROWS: 3

-- Q242: CURTIME function (alias for CURRENT_TIME)
SELECT * FROM (SELECT id, CURTIME() as time_now FROM organizations LIMIT 3);
-- EXPECT_ROWS: 3

-- Q243: SYSDATE function
SELECT * FROM (SELECT id, SYSDATE() as sys_now FROM organizations LIMIT 3);
-- EXPECT_ROWS: 3

-- Q244: LOCALTIME function
SELECT * FROM (SELECT id, LOCALTIME() as local_time FROM organizations LIMIT 3);
-- EXPECT_ROWS: 3

-- Q245: LOCALTIMESTAMP function
SELECT * FROM (SELECT id, LOCALTIMESTAMP() as local_ts FROM organizations LIMIT 3);
-- EXPECT_ROWS: 3

-- Q246: DAYOFMONTH function (alias for DAY)
SELECT * FROM (SELECT id, creation_date, DAYOFMONTH(creation_date) as dom FROM organizations LIMIT 5);
-- EXPECT_ROWS: 5

-- Q247: WEEKOFYEAR function (alias for WEEK)
SELECT * FROM (SELECT id, creation_date, WEEKOFYEAR(creation_date) as woy FROM organizations LIMIT 5);
-- EXPECT_ROWS: 5

-- Q248: ADDDATE function (alias for DATE_ADD)
SELECT * FROM (SELECT id, creation_date, ADDDATE(creation_date, 7) as plus_week FROM organizations LIMIT 5);
-- EXPECT_ROWS: 5

-- Q249: SUBDATE function (alias for DATE_SUB)
SELECT * FROM (SELECT id, creation_date, SUBDATE(creation_date, 7) as minus_week FROM organizations LIMIT 5);
-- EXPECT_ROWS: 5

-- Q250: ADDTIME function
SELECT * FROM (SELECT id, ADDTIME('10:00:00', '01:30:00') as added_time FROM organizations LIMIT 3);
-- EXPECT_ROWS: 3

-- Q251: SUBTIME function
SELECT * FROM (SELECT id, SUBTIME('12:00:00', '02:30:00') as sub_time FROM organizations LIMIT 3);
-- EXPECT_ROWS: 3

-- Q252: TIMEDIFF function
SELECT * FROM (SELECT id, TIMEDIFF('14:30:00', '10:00:00') as time_diff FROM organizations LIMIT 3);
-- EXPECT_ROWS: 3

-- Q253: PERIOD_ADD function
SELECT * FROM (SELECT id, PERIOD_ADD(202401, 3) as new_period FROM organizations LIMIT 3);
-- EXPECT_ROWS: 3

-- Q254: PERIOD_DIFF function
SELECT * FROM (SELECT id, PERIOD_DIFF(202406, 202401) as months_diff FROM organizations LIMIT 3);
-- EXPECT_ROWS: 3

-- Q255: DATE_FORMAT function
SELECT * FROM (SELECT id, creation_date, DATE_FORMAT('%Y-%m-%d', creation_date) as formatted FROM organizations LIMIT 5);
-- EXPECT_ROWS: 5

-- Q256: DATE_FORMAT with custom pattern
SELECT * FROM (SELECT id, creation_date, DATE_FORMAT('%M %D, %Y', creation_date) as formatted FROM organizations LIMIT 5);
-- EXPECT_ROWS: 5

-- Q257: TIME_FORMAT function
SELECT * FROM (SELECT id, TIME_FORMAT('%H:%i:%s', '14:30:45') as formatted FROM organizations LIMIT 3);
-- EXPECT_ROWS: 3

-- Q258: STRFTIME function (alias for DATE_FORMAT)
SELECT * FROM (SELECT id, creation_date, STRFTIME('%Y/%m/%d', creation_date) as formatted FROM organizations LIMIT 5);
-- EXPECT_ROWS: 5

-- Q259: STR_TO_DATE function
SELECT * FROM (SELECT id, STR_TO_DATE('2024-06-15', '%Y-%m-%d') as parsed FROM organizations LIMIT 3);
-- EXPECT_ROWS: 3

-- Q260: MICROSECOND function
SELECT * FROM (SELECT id, MICROSECOND('10:30:45.123456') as micros FROM organizations LIMIT 3);
-- EXPECT_ROWS: 3

-- =============================================================================
-- SECTION 29: DateTime Functions in WHERE Clause
-- =============================================================================

-- Q261: Filter by YEAR
SELECT * FROM (SELECT id, name, creation_date FROM organizations WHERE YEAR(creation_date) >= 2020 LIMIT 10);
-- EXPECT_ROWS: >= 0

-- Q262: Filter by MONTH
SELECT * FROM (SELECT id, name, creation_date FROM organizations WHERE MONTH(creation_date) = 6 LIMIT 10);
-- EXPECT_ROWS: >= 0

-- Q263: Filter by DAY
SELECT * FROM (SELECT id, name, creation_date FROM organizations WHERE DAY(creation_date) = 15 LIMIT 10);
-- EXPECT_ROWS: >= 0

-- Q264: Filter by QUARTER
SELECT * FROM (SELECT id, name, creation_date FROM organizations WHERE QUARTER(creation_date) = 2 LIMIT 10);
-- EXPECT_ROWS: >= 0

-- Q265: Filter by DAYOFWEEK (Sunday=1)
SELECT * FROM (SELECT id, name, creation_date FROM organizations WHERE DAYOFWEEK(creation_date) = 1 LIMIT 10);
-- EXPECT_ROWS: >= 0

-- Q266: Filter by WEEKDAY (Monday=0)
SELECT * FROM (SELECT id, name, creation_date FROM organizations WHERE WEEKDAY(creation_date) = 0 LIMIT 10);
-- EXPECT_ROWS: >= 0

-- Q267: Filter by DAYNAME
SELECT * FROM (SELECT id, name, creation_date, DAYNAME(creation_date) as dow FROM organizations WHERE DAYNAME(creation_date) = 'Saturday' LIMIT 10);
-- EXPECT_ROWS: >= 0

-- Q268: Filter by MONTHNAME
SELECT * FROM (SELECT id, name, creation_date, MONTHNAME(creation_date) as mon FROM organizations WHERE MONTHNAME(creation_date) = 'June' LIMIT 10);
-- EXPECT_ROWS: >= 0

-- Q269: Filter by WEEK number
SELECT * FROM (SELECT id, name, creation_date FROM organizations WHERE WEEK(creation_date) = 25 LIMIT 10);
-- EXPECT_ROWS: >= 0

-- Q270: Filter by DATEDIFF (records older than 365 days)
SELECT * FROM (SELECT id, name, creation_date, DATEDIFF(NOW(), creation_date) as days_old FROM organizations WHERE DATEDIFF(NOW(), creation_date) > 365 LIMIT 10);
-- EXPECT_ROWS: >= 0

-- Q271: Filter by TO_DAYS comparison
SELECT * FROM (SELECT id, name, creation_date FROM organizations WHERE TO_DAYS(creation_date) > 738000 LIMIT 10);
-- EXPECT_ROWS: >= 0

-- Q272: Filter by LAST_DAY (last day of month)
SELECT * FROM (SELECT id, name, creation_date FROM organizations WHERE DAY(creation_date) = DAY(LAST_DAY(creation_date)) LIMIT 10);
-- EXPECT_ROWS: >= 0

-- Q273: Filter using DATE_ADD
SELECT * FROM (SELECT id, name, creation_date FROM organizations WHERE DATE_ADD(creation_date, 365) < NOW() LIMIT 10);
-- EXPECT_ROWS: >= 0

-- Q274: Filter using DATE_SUB
SELECT * FROM (SELECT id, name, creation_date FROM organizations WHERE creation_date > DATE_SUB(NOW(), 730) LIMIT 10);
-- EXPECT_ROWS: >= 0

-- Q275: Combined date filters (year AND month)
SELECT * FROM (SELECT id, name, creation_date FROM organizations WHERE YEAR(creation_date) = 2023 AND MONTH(creation_date) >= 6 LIMIT 10);
-- EXPECT_ROWS: >= 0

-- Q276: Combined date filters with OR
SELECT * FROM (SELECT id, name, creation_date FROM organizations WHERE QUARTER(creation_date) = 1 OR QUARTER(creation_date) = 4 LIMIT 10);
-- EXPECT_ROWS: >= 0

-- Q277: Filter by DAYOFYEAR
SELECT * FROM (SELECT id, name, creation_date, DAYOFYEAR(creation_date) as doy FROM organizations WHERE DAYOFYEAR(creation_date) > 180 LIMIT 10);
-- EXPECT_ROWS: >= 0

-- Q278: Filter by YEARWEEK
SELECT * FROM (SELECT id, name, creation_date, YEARWEEK(creation_date) as yw FROM organizations WHERE YEARWEEK(creation_date) >= 202301 LIMIT 10);
-- EXPECT_ROWS: >= 0

-- Q279: Complex date calculation in WHERE
SELECT * FROM (SELECT id, name, creation_date FROM organizations WHERE YEAR(creation_date) * 100 + MONTH(creation_date) >= 202306 LIMIT 10);
-- EXPECT_ROWS: >= 0

-- Q280: Filter by date expression result
SELECT * FROM (SELECT id, name, creation_date, DATEDIFF(NOW(), creation_date) / 365 as years FROM organizations WHERE DATEDIFF(NOW(), creation_date) / 365 > 1 LIMIT 10);
-- EXPECT_ROWS: >= 0

-- =============================================================================
-- SECTION 30: DateTime with GROUP BY
-- =============================================================================

-- Q281: GROUP BY YEAR
SELECT * FROM (SELECT YEAR(creation_date) as year, COUNT(*) as cnt FROM organizations GROUP BY YEAR(creation_date) ORDER BY year DESC LIMIT 10);
-- EXPECT_ROWS: >= 1

-- Q282: GROUP BY MONTH
SELECT * FROM (SELECT MONTH(creation_date) as month, COUNT(*) as cnt FROM organizations GROUP BY MONTH(creation_date) ORDER BY month LIMIT 12);
-- EXPECT_ROWS: >= 1

-- Q283: GROUP BY QUARTER
SELECT * FROM (SELECT QUARTER(creation_date) as quarter, COUNT(*) as cnt FROM organizations GROUP BY QUARTER(creation_date) ORDER BY quarter);
-- EXPECT_ROWS: >= 1

-- Q284: GROUP BY DAYOFWEEK
SELECT * FROM (SELECT DAYOFWEEK(creation_date) as dow, COUNT(*) as cnt FROM organizations GROUP BY DAYOFWEEK(creation_date) ORDER BY dow);
-- EXPECT_ROWS: >= 1

-- Q285: GROUP BY YEAR and MONTH
SELECT * FROM (SELECT YEAR(creation_date) as year, MONTH(creation_date) as month, COUNT(*) as cnt FROM organizations GROUP BY YEAR(creation_date), MONTH(creation_date) ORDER BY year DESC, month DESC LIMIT 10);
-- EXPECT_ROWS: >= 1

-- Q286: GROUP BY YEARWEEK
SELECT * FROM (SELECT YEARWEEK(creation_date) as yw, COUNT(*) as cnt FROM organizations GROUP BY YEARWEEK(creation_date) ORDER BY yw DESC LIMIT 10);
-- EXPECT_ROWS: >= 1

-- Q287: GROUP BY DAYNAME
SELECT * FROM (SELECT DAYNAME(creation_date) as day_name, COUNT(*) as cnt FROM organizations GROUP BY DAYNAME(creation_date) ORDER BY cnt DESC);
-- EXPECT_ROWS: >= 1

-- Q288: GROUP BY MONTHNAME
SELECT * FROM (SELECT MONTHNAME(creation_date) as month_name, COUNT(*) as cnt FROM organizations GROUP BY MONTHNAME(creation_date) ORDER BY cnt DESC LIMIT 12);
-- EXPECT_ROWS: >= 1

-- Q289: Aggregate with date functions
SELECT * FROM (SELECT MIN(YEAR(creation_date)) as min_year, MAX(YEAR(creation_date)) as max_year, COUNT(DISTINCT YEAR(creation_date)) as years_count FROM organizations);
-- EXPECT_ROWS: 1

-- Q290: Date range stats
SELECT * FROM (SELECT YEAR(creation_date) as year, MIN(creation_date) as first_date, MAX(creation_date) as last_date, COUNT(*) as cnt FROM organizations GROUP BY YEAR(creation_date) ORDER BY year DESC LIMIT 5);
-- EXPECT_ROWS: >= 1

-- =============================================================================
-- SECTION 31: DateTime with ORDER BY
-- =============================================================================

-- Q291: ORDER BY YEAR DESC
SELECT * FROM (SELECT id, name, creation_date, YEAR(creation_date) as year FROM organizations ORDER BY YEAR(creation_date) DESC, id LIMIT 10);
-- EXPECT_ROWS: 10

-- Q292: ORDER BY MONTH ASC
SELECT * FROM (SELECT id, name, creation_date, MONTH(creation_date) as month FROM organizations ORDER BY MONTH(creation_date) ASC LIMIT 10);
-- EXPECT_ROWS: 10

-- Q293: ORDER BY DATEDIFF (oldest first)
SELECT * FROM (SELECT id, name, creation_date, DATEDIFF(NOW(), creation_date) as age_days FROM organizations ORDER BY DATEDIFF(NOW(), creation_date) DESC LIMIT 10);
-- EXPECT_ROWS: 10

-- Q294: ORDER BY DAYOFYEAR
SELECT * FROM (SELECT id, name, creation_date, DAYOFYEAR(creation_date) as doy FROM organizations ORDER BY DAYOFYEAR(creation_date) LIMIT 10);
-- EXPECT_ROWS: 10

-- Q295: ORDER BY QUARTER then MONTH
SELECT * FROM (SELECT id, name, creation_date, QUARTER(creation_date) as q, MONTH(creation_date) as m FROM organizations ORDER BY QUARTER(creation_date), MONTH(creation_date) LIMIT 10);
-- EXPECT_ROWS: 10

-- =============================================================================
-- SECTION 32: DateTime with Subqueries
-- =============================================================================

-- Q296: Subquery with date filter
SELECT * FROM (SELECT * FROM (SELECT id, name, creation_date FROM organizations WHERE YEAR(creation_date) >= 2020) AS sub LIMIT 10);
-- EXPECT_ROWS: >= 0

-- Q297: Nested date calculations
SELECT * FROM (SELECT * FROM (SELECT id, name, creation_date, YEAR(creation_date) as y, QUARTER(creation_date) as q FROM organizations) AS sub WHERE y >= 2020 AND q >= 2 LIMIT 10);
-- EXPECT_ROWS: >= 0

-- Q298: Date aggregate in subquery
SELECT * FROM (SELECT year, total FROM (SELECT YEAR(creation_date) as year, COUNT(*) as total FROM organizations GROUP BY YEAR(creation_date)) AS sub ORDER BY year DESC LIMIT 5);
-- EXPECT_ROWS: >= 1

-- Q299: Multiple date functions in subquery
SELECT * FROM (SELECT id, creation_date, CONCAT(YEAR(creation_date), '-', LPAD(MONTH(creation_date), 2, '0')) as year_month, DAYNAME(creation_date) as day_name FROM organizations LIMIT 10);
-- EXPECT_ROWS: 10

-- Q300: Date difference comparison in nested query
SELECT * FROM (SELECT * FROM (SELECT id, name, creation_date, DATEDIFF(NOW(), creation_date) as days_old FROM organizations) AS sub WHERE days_old > 100 ORDER BY days_old DESC LIMIT 10);
-- EXPECT_ROWS: >= 0
